use scirust::attention::slha_v2::{SciRustSlhaTile, D_C, RESIDUAL_WORDS};
use scirust::audit;
use std::panic::catch_unwind;
use std::ptr;

/// Opaque handle for the SLHA context (if needed in future, currently stateless).
#[repr(C)]
pub struct SlhaContext {
    _unused: [u8; 0],
}

/// Initialize the SLHA environment.
#[no_mangle]
pub extern "C" fn slha_init() -> *mut SlhaContext {
    // Currently stateless, but providing the entry point for future-proofing.
    // Returning a dummy non-null pointer for now.
    static mut DUMMY: u8 = 0;
    std::ptr::addr_of_mut!(DUMMY) as *mut SlhaContext
}

/// Process a single tile and compute the score.
///
/// # Safety
/// `tile`, `q_coarse`, and `q_sign` must be valid non-null pointers.
/// `q_coarse` must point to an array of `D_C` (128) f32s.
/// `q_sign` must point to an array of `RESIDUAL_WORDS` (4) u64s.
#[no_mangle]
pub unsafe extern "C" fn slha_process_tile(
    tile: *const SciRustSlhaTile,
    q_coarse: *const f32,
    q_sign: *const u64,
    score_out: *mut f32,
) -> i32 {
    let result = catch_unwind(|| {
        if tile.is_null() || q_coarse.is_null() || q_sign.is_null() || score_out.is_null() {
            return -1;
        }

        let tile_ref = &*tile;
        let q_coarse_ref = &*(q_coarse as *const [f32; D_C]);
        let q_sign_ref = &*(q_sign as *const [u64; RESIDUAL_WORDS]);

        *score_out = tile_ref.compute_score(q_coarse_ref, q_sign_ref);
        0
    });

    result.unwrap_or(-2) // -2 if panic occurred
}

/// Run the self-audit and return a JSON string.
///
/// # Safety
/// The returned pointer must be freed using `slha_free_string`.
#[no_mangle]
pub unsafe extern "C" fn slha_audit() -> *mut i8 {
    let result = catch_unwind(|| {
        let report = audit::run();
        let s = report.to_compact();
        std::ffi::CString::new(s).unwrap().into_raw()
    });

    result.unwrap_or(ptr::null_mut())
}

/// Free a string allocated by the library.
///
/// # Safety
/// `s` must be a pointer returned by `slha_audit`.
#[no_mangle]
pub unsafe extern "C" fn slha_free_string(s: *mut i8) {
    if !s.is_null() {
        let _ = std::ffi::CString::from_raw(s);
    }
}
