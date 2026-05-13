//! World text unlit (Unity shader asset `TextUnit`): MSDF / SDF / raster font atlas in world space.
//!
//! Compatibility route that shares the `textunlit` shader body. The host resolves
//! `TextUnit` to `textunit_default` / `textunit_multiview`; the source alias below reuses
//! `textunlit.wgsl` directly.
//#source_alias textunlit
