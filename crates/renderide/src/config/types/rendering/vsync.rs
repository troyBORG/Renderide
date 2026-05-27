//! Swapchain vsync mode (`[rendering] vsync`).

use crate::labeled_enum;

labeled_enum! {
    /// Swapchain vsync mode persisted in `config.toml` as `[rendering] vsync`.
    ///
    /// Two user-facing values matching what desktop and VR titles typically expose: **Off**
    /// (tearing, lowest latency) and **On** (strict vblank pacing through `Fifo`). Defaults to
    /// [`Self::Off`].
    ///
    /// Resolution to a [`wgpu::PresentMode`] happens in [`VsyncMode::resolve_present_mode`],
    /// which probes the surface's actual capabilities instead of passing wgpu's `Auto*` shortcuts
    /// through to surface configuration.
    ///
    /// The bool-shape branch lets the historical `vsync = true / false` syntax keep loading
    /// without manual migration; the alias list further covers removed
    /// `vsync = "auto"` / `"adaptive"` / `"fifo_relaxed"` tokens.
    pub enum VsyncMode: "vsync mode (`off` / `on`)" {
        default    => Off;
        bool_true  => On;
        bool_false => Off;

        /// No vsync. Lowest latency, may tear; CPU/GPU run uncapped. Resolves to `Immediate`
        /// when the surface advertises it, otherwise falls through `Mailbox` and finally `Fifo`.
        Off => {
            persist: "off",
            label: "Off",
            aliases: ["false", "0", "no", "none"],
        },
        /// Strict vsync. Resolves to `Fifo` so presentation stays vblank-paced and does not
        /// intentionally tear when a frame misses its deadline.
        On => {
            persist: "on",
            label: "On",
            aliases: [
                "true",
                "1",
                "yes",
                "vsync",
                "fifo",
                "auto",
                "adaptive",
                "fifo_relaxed",
                "fiforelaxed",
                "relaxed",
            ],
        },
    }
}

impl VsyncMode {
    /// Resolves this mode to a [`wgpu::PresentMode`] that the surface actually supports, using
    /// explicit preference chains rather than wgpu's lazy `Auto*` shortcuts.
    ///
    /// Each variant walks an ordered preference list and picks the first entry present in
    /// `supported` ([`wgpu::SurfaceCapabilities::present_modes`]). [`wgpu::PresentMode::Fifo`]
    /// is required to be supported by every conformant surface ([wgpu spec][1]), so the chain
    /// always terminates.
    ///
    /// | Variant            | Preference order                            | Behavior                                                          |
    /// | ------------------ | ------------------------------------------- | ----------------------------------------------------------------- |
    /// | [`Self::Off`]      | `Immediate` -> `Mailbox` -> `Fifo`            | Lowest latency; tears                                             |
    /// | [`Self::On`]       | `Fifo`                                      | Vblank-paced, no intentional tearing                              |
    ///
    /// Compatibility aliases such as `auto`, `adaptive`, and `fifo_relaxed` still load as
    /// [`Self::On`], but the runtime behavior is strict FIFO.
    ///
    /// [1]: https://www.w3.org/TR/webgpu/#dom-gpupresentmode-fifo
    pub fn resolve_present_mode(self, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
        use wgpu::PresentMode::{Fifo, Immediate, Mailbox};
        match self {
            Self::Off => first_supported_present_mode(&[Immediate, Mailbox, Fifo], supported),
            Self::On => first_supported_present_mode(&[Fifo], supported),
        }
    }
}

/// Walks `preferred` in order and returns the first variant present in `supported`, falling
/// back to [`wgpu::PresentMode::Fifo`] when nothing matches.
///
/// `Fifo` is the unconditional fallback because every conformant surface advertises it; see
/// [`VsyncMode::resolve_present_mode`] for the per-mode preference chains that route through here.
fn first_supported_present_mode(
    preferred: &[wgpu::PresentMode],
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    preferred
        .iter()
        .copied()
        .find(|m| supported.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

#[cfg(test)]
mod tests {
    use super::VsyncMode;
    use crate::config::types::RendererSettings;
    use wgpu::PresentMode;

    #[test]
    fn off_prefers_immediate_when_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
    }

    #[test]
    fn modes_choose_preferred_modes_when_everything_is_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Fifo
        );
    }

    #[test]
    fn off_falls_through_to_mailbox_then_fifo() {
        let mailbox_only = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&mailbox_only),
            PresentMode::Mailbox
        );
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn on_uses_fifo_even_when_relaxed_or_mailbox_supported() {
        let supported = [
            PresentMode::Mailbox,
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
        ];
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Fifo
        );
    }

    #[test]
    fn empty_supported_list_falls_back_to_fifo() {
        for mode in VsyncMode::ALL.iter().copied() {
            assert_eq!(
                mode.resolve_present_mode(&[]),
                PresentMode::Fifo,
                "mode {mode:?} must terminate at Fifo when nothing is advertised"
            );
        }
    }

    #[test]
    fn legacy_auto_and_adaptive_tokens_load_as_on() {
        for token in ["auto", "adaptive", "fifo_relaxed", "fiforelaxed", "relaxed"] {
            let toml = format!("[rendering]\nvsync = \"{token}\"\n");
            let parsed: RendererSettings = toml::from_str(&toml).expect("legacy vsync alias");
            assert_eq!(
                parsed.rendering.vsync,
                VsyncMode::On,
                "token `{token}` must map to On"
            );
        }
    }

    #[test]
    fn legacy_boolean_shape_loads() {
        let on: RendererSettings =
            toml::from_str("[rendering]\nvsync = true\n").expect("bool true");
        assert_eq!(on.rendering.vsync, VsyncMode::On);
        let off: RendererSettings =
            toml::from_str("[rendering]\nvsync = false\n").expect("bool false");
        assert_eq!(off.rendering.vsync, VsyncMode::Off);
    }

    #[test]
    fn on_serializes_as_on() {
        let mut s = RendererSettings::default();
        s.rendering.vsync = VsyncMode::On;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(back.rendering.vsync, VsyncMode::On);
        assert!(
            toml.contains("vsync = \"on\""),
            "expected `on` in serialized TOML, got: {toml}"
        );
    }
}
