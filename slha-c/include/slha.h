#ifndef SLHA_H
#define SLHA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLHA_D_C 128
#define SLHA_D_S 256
#define SLHA_LATENT_BYTES 64
#define SLHA_RESIDUAL_WORDS 4
#define SLHA_N_GROUPS 8

/**
 * SLHA v2 Tile structure.
 *
 * Alignment: 64 or 128 bytes depending on the platform's cache line.
 * Total size: exactly 128 bytes.
 */
#if defined(__GNUC__) || defined(__clang__)
#define SLHA_ALIGN_64 __attribute__((aligned(64)))
#else
#define SLHA_ALIGN_64
#endif

typedef struct SLHA_ALIGN_64 {
    uint8_t latent_kv[SLHA_LATENT_BYTES];
    uint64_t residual_bitmap[SLHA_RESIDUAL_WORDS];
    float scale;
    float dynamic_lambda;
    float residual_sigma;
    uint32_t token_id;
    uint32_t position;
    uint16_t head_id;
    uint16_t flags;
    uint8_t group_scales[SLHA_N_GROUPS];
} SciRustSlhaTile;

typedef struct SlhaContext SlhaContext;

/**
 * Initialize the SLHA environment.
 */
SlhaContext* slha_init();

/**
 * Process a single tile and compute the score.
 *
 * Returns 0 on success, negative values on error.
 */
int32_t slha_process_tile(
    const SciRustSlhaTile* tile,
    const float* q_coarse,
    const uint64_t* q_sign,
    float* score_out
);

/**
 * Run the self-audit and return a JSON string.
 * The caller must free the string using slha_free_string.
 */
char* slha_audit();

/**
 * Free a string allocated by the library.
 */
void slha_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* SLHA_H */
