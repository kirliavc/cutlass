/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * A block-wide task abstraction for computing device-wide GEMM
 */

#pragma once

// Compiler guard conditional to avoid compilation errors on versions of CUDA that
// do not support the WMMA API.
#if defined (WMMA)

#include <stdint.h>

#include "../util/util.h"

#include "grid_raster.h"
#include "block_loader.h"
#include "block_loader_wmma.h"
#include "wmma_accumulator.h"
#include <mma.h>

namespace cutlass {
namespace gemm {


/******************************************************************************
 * block_task_wmma_policy
 ******************************************************************************/

/**
 * \brief Parameterizable tuning policy for block-wide WMMA GEMM tasks
 *
 * Once parameterized, \p block_task_policy provides the member constant
 * \p BlockThreads indicating to the required thread block size
 */
template <
    int _BlockItemsY,                               ///< Height in rows of a block-wide tile in matrix C
    int _BlockItemsX,                               ///< Width in columns of a block-wide tile in matrix C
    int _BlockItemsK,                               ///< Extent of block-wide A|B tiles in value_t along the K-axis
    int _WarpItemsY,                                ///< Height in rows of a Warp tile's accumulators
    int _WarpItemsX,                                ///< Width in columns of a Warp tile's accumulators
    int _WmmaItemsY,                                ///< Height in rows of a discrete WMMA block's accumulators
    int _WmmaItemsX,                                ///< Width in columns of a discrete WMMA block's accumulators
    int _WmmaItemsK,                                ///< Depth of each discrete WMMA block
    bool _UseDoubleScratchTiles,                    ///< Whether to halve synchronization overhead at the expense of doubled shared memory and addressing overhead
    grid_raster_strategy::kind_t _RasterStrategy>   ///< Strategy for enumerating \p block_task within an input matrix
struct block_task_wmma_policy
{
    /// Strategy for enumerating \p block_task within an input matrix
    static const grid_raster_strategy::kind_t RasterStrategy = _RasterStrategy;

    enum
    {
        /// Height in rows of a block-wide tile in matrix C
        BlockItemsY = _BlockItemsY,

        /// Width in columns of a block-wide tile in matrix C
        BlockItemsX = _BlockItemsX,

        /// Extent of block-wide A|B tiles in value_t along the K-axis
        BlockItemsK = _BlockItemsK,

        /// Height in rows of a Warp tile's accumulators
        WarpItemsX = _WarpItemsX,

        /// Width in columns of a Warp tile's accumulators
        WarpItemsY = _WarpItemsY,

        /// Width in columns of a discrete WMMA block's accumulators
        WmmaItemsX = _WmmaItemsX,

        /// Height in rows of a discrete WMMA block's accumulators
        WmmaItemsY = _WmmaItemsY,

        /// Depth of each discrete WMMA block
        WmmaItemsK = _WmmaItemsK,

        /// Whether to halve synchronization overhead at the expense of doubled shared memory and addressing overhead
        UseDoubleScratchTiles = _UseDoubleScratchTiles,


        //
        // Derived quantities
        //

        /// Machine warp size
        WarpThreads = 32,

        /// Number of WMMA operations in the height dimension
        WmmaBlocksY = divide_assert<WarpItemsY, WmmaItemsY>::value,

        /// Number of WMMA operations in the height dimension
        WmmaBlocksX = divide_assert<WarpItemsX, WmmaItemsX>::value,

        /// Number of warps in each thread block
        BlockWarps = divide_assert<BlockItemsY * BlockItemsX, WarpItemsX * WarpItemsY>::value,

        /// Number of threads in each thread block (blockDim.x)
        BlockThreads = BlockWarps * WarpThreads,
    };
};


/******************************************************************************
 * block_task_wmma
 ******************************************************************************/

/**
 * \brief A block-wide task abstraction for computing device-wide GEMM
 *
 * Each thread_block is assigned a unique tile of output matrix C to compute by
 * consuming the corresponding stripes of the input matrices A and B.
 */
template <
    typename                    block_task_policy_t,    ///< Parameterization of block_task_policy
    typename                    value_t,                ///< Multiplicand value type (matrices A and B)
    typename                    accum_t,                ///< Accumulator value type (matrix C and scalars)
    matrix_transform_t::kind_t  TransformA,             ///< View transform enumerant for matrix A
    int                         LdgAlignA,              ///< Alignment (in bytes) for A operand
    matrix_transform_t::kind_t  TransformB,             ///< View transform enumerant for matrix B
    int                         LdgAlignB,              ///< Alignment (in bytes) for B operand
    typename                    epilogue_op_t,          ///< Epilogue operation to update matrix C
    int                         LdgAlignC,              ///< Alignment (in bytes) for C operand
    bool                        AllowRaggedTiles        ///< Whether the input matrix's dimensions need not be an even-multiple of the block-wide tile dimensions
>
struct block_task_wmma
{
    //-------------------------------------------------------------------------
    // Constants and types
    //-------------------------------------------------------------------------

    enum
    {
        /// Number of threads in each thread block (blockDim.x)
        BlockThreads = block_task_policy_t::BlockThreads,

        /// Extent of block-wide C-tile in accum_t (and A-tiles in value_t) along M-axis (height)
        BlockItemsY = block_task_policy_t::BlockItemsY,

        /// Extent of block-wide C-tile in accum_t (and B-tiles in value_t) along N-axis (width)
        BlockItemsX = block_task_policy_t::BlockItemsX,

        /// Extent of block-wide A|B tiles in value_t along the K-axis
        BlockItemsK = block_task_policy_t::BlockItemsK,

        /// Extent of warp C-tile in accum_t (and A-tiles in value_t) along M-axis (height)
        WarpItemsY = block_task_policy_t::WarpItemsY,

        /// Extent of warp C-tile in accum_t (and B-tiles in value_t) along N-axis (width)
        WarpItemsX = block_task_policy_t::WarpItemsX,

        /// Extent of warp C-tile in accum_t (and A-tiles in value_t) along M-axis (height)
        WmmaItemsY = block_task_policy_t::WmmaItemsY,

        /// Extent of warp C-tile in accum_t (and B-tiles in value_t) along N-axis (width)
        WmmaItemsX = block_task_policy_t::WmmaItemsX,

        /// Extent of warp-wide A|B-tiles in value_t along K-axis
        WmmaItemsK = block_task_policy_t::WmmaItemsK,

        /// Whether to halve synchronization overhead at the expense of doubled shared memory and addressing overhead
        UseDoubleScratchTiles = block_task_policy_t::UseDoubleScratchTiles,

        /// Number of threads in warp
        WarpThreads = block_task_policy_t::WarpThreads,

        /// Number of warps participating
        BlockWarps = block_task_policy_t::BlockWarps,

        /// Extent of block in warps along M-axis
        BlockWarpsY = divide_assert<BlockItemsY, WarpItemsY>::value,

        /// Extent of block in warps along N-axis
        BlockWarpsX = divide_assert<BlockItemsX, WarpItemsX>::value,

        /// Number of MMA unrolls
        WmmaUnrollCount = divide_assert<BlockItemsK,  WmmaItemsK>::value,

        /// True if the A matrix layout is column major (K is the strided dimension)
        IsLayoutCongruousA = (TransformA == matrix_transform_t::NonTranspose),

        /// True if the B matrix layout is row mayor (K is the strided dimension)
        IsLayoutCongruousB = (TransformB == matrix_transform_t::Transpose),

        ThreadItemsX = 4,

        ThreadItemsY = 2,

        /// Number of WMMA blocks in warp row
        WmmaBlocksX = divide_assert<WarpItemsX, WmmaItemsX>::value,

        /// Number of WMMA blocks in a warp column
        WmmaBlocksY = divide_assert<WarpItemsY, WmmaItemsY>::value,
    };

    /// WMMA may support unique types for A and B, so plan ahead for this
    typedef value_t value_a_t;

    /// WMMA may support unique types for A and B, so plan ahead for this
    typedef value_t value_b_t;

    /// Thread block rasterization helper type
    typedef grid_raster<
            BlockItemsY,
            BlockItemsX,
            TransformA,
            TransformB,
            block_task_policy_t::RasterStrategy>
        grid_raster_t;

    /// Tile loader type for matrix A
    typedef block_loader_wmma<
            IsLayoutCongruousA,
            BlockThreads,
            (IsLayoutCongruousA ? BlockItemsY : BlockItemsK),
            (IsLayoutCongruousA ? BlockItemsK : BlockItemsY),
            value_a_t,
            LdgAlignA,
            AllowRaggedTiles>
        block_loader_a_t;

    /// Tile loader type for matrix A
    typedef block_loader_wmma<
            IsLayoutCongruousB,
            BlockThreads,
            (IsLayoutCongruousB ? BlockItemsX : BlockItemsK),
            (IsLayoutCongruousB ? BlockItemsK : BlockItemsX),
            value_b_t,
            LdgAlignB,
            AllowRaggedTiles>
        block_loader_b_t;


/// Fragment type for matrix operand A
    typedef nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a,
            WmmaItemsY,
            WmmaItemsX,
            WmmaItemsK,
            value_a_t,
            typename matrix_layout<TransformA>::tag>
        fragment_a_t;

    /// Fragment type for matrix operand B
    typedef nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b,
            WmmaItemsY,
            WmmaItemsX,
            WmmaItemsK,
            value_b_t,
            typename matrix_layout<TransformB>::tag>
        fragment_b_t;

    /// Fragment type for accumulator
    typedef nvcuda::wmma::fragment<
            nvcuda::wmma::accumulator,
            WmmaItemsY,
            WmmaItemsX,
            WmmaItemsK,
            accum_t>
        accumulator_t;

    typedef thread_accumulator<
            ThreadItemsY,
            ThreadItemsX,
            value_t,
            accum_t>
        thread_accumulator_t;

    /// Dot-product vector type along the K-axis (e.g, uchar4 when using IDP4A)
    typedef typename thread_accumulator_t::dp_vector_t dp_vector_t;

    enum
    {
        /// Number of fragments from A matrix

        /// Number of value_t to pad the outer dimension of the shared A-tile
        PadItemsA = 16,

        /// Number of value_t to pad the outer dimension of the shared B-tile
        PadItemsB = 16,

        /// Leading dimension of A matrix tile
        LdmSmemA = (IsLayoutCongruousA ? BlockItemsY: BlockItemsK) + PadItemsA,

        /// Leading dimension of A matrix tile
        StridedSmemA = (IsLayoutCongruousA ? BlockItemsK : BlockItemsY ),

        /// Leading dimension of B matrix tile
        LdmSmemB = (IsLayoutCongruousB? BlockItemsX : BlockItemsK)  + PadItemsB,

        StridedSmemB = (IsLayoutCongruousB ? BlockItemsK : BlockItemsX),

        LdsVectorDpVectorsA = __NV_STD_MIN(
            ThreadItemsY, 
            __NV_STD_MAX(1, (128 / (__NV_STD_MAX(sizeof(dp_vector_t), sizeof(accum_t)) * 8)))),

        /// Number of dp_vector_t along N-axis that can be read in a single LDS from the shared B-tile (up to 128b if more than one value_t)
        LdsVectorDpVectorsB = __NV_STD_MIN(
            ThreadItemsX, 
            __NV_STD_MAX(1, (128 / (__NV_STD_MAX(sizeof(dp_vector_t), sizeof(accum_t)) * 8)))),


        /// Number of strip-mined LDS vector reads from shared A-tile
        ThreadLdsVectorsA = divide_assert<ThreadItemsY, LdsVectorDpVectorsA>::value,

        /// Number of strip-mined LDS vector reads from shared B-tile
        ThreadLdsVectorsB = divide_assert<ThreadItemsX, LdsVectorDpVectorsB>::value,

        WarpThreadsX = 4,

        WarpThreadsY = 8,
    };

        /// Load-from-shared data movement type for A-tile, coarsened by LdsVectorDpVectorsA
    typedef io_vector<dp_vector_t, LdsVectorDpVectorsA> lds_vector_a_t;

    /// Load-from-shared data movement type for B-tile, coarsened by LdsVectorDpVectorsB
    typedef io_vector<dp_vector_t, LdsVectorDpVectorsB> lds_vector_b_t;
    /// Shared memory layout for a prefetch page
    struct page_storage_t
    {
        /// Tile of A
        value_a_t __align__(16) block_a[StridedSmemA][LdmSmemA];

        /// Tile of B
        value_b_t __align__(16) block_b[StridedSmemB][LdmSmemB];
    };

    /// Shared memory layout for scratch storage
    struct scratch_storage_t
    {
        union
        {
            /// Prefetch pages
            uninitialized<page_storage_t> pages[UseDoubleScratchTiles ? 2 : 1];

            /// Scratch storage for warps
            accum_t epilogue[BlockWarps][WmmaItemsX * WmmaItemsY];
        };
    };

    //-------------------------------------------------------------------------
    // Assert assumptions
    //-------------------------------------------------------------------------

    // Ensure we have at least two unrolled innermost loop iterations (one to prefetch
    // the next global tile and then one to prefetch the first strip of it from shared)
    static_assert ((BlockItemsK >= 2), "BlockItemsK must be >= 2.");

    //-------------------------------------------------------------------------
    // Members
    //-------------------------------------------------------------------------

    /// Scratch storage reference
    scratch_storage_t *scratch;

    /// Which page of scratch tiles we're currently reading from
    int page_idx;

    /// Pointer to matrix C
    accum_t *d_c;

    /// Epilogue operation applied to update matrix C
    epilogue_op_t epilogue_op;

    /// Matrix height in rows of trans_op(A) and C
    int dim_m;

    /// Matrix width in columns of trans_op(B) and C
    int dim_n;

    /// Control for inter-block k-splitting
    k_split_control k_split;

    /// Thread block's base value_t coordinates (m, n) in matrix C
    grid_raster_t grid_raster;

    /// Thread block's current coordinate (k) within A|B matrices
    int block_item_coords_k;

    /// Thread block's ending coordinate (k) within A|B matrices (one-past)
    int block_end_item_k;

    /// Warp's coordinates (x, y) in thread block
    int2 block_warp_item_coords;

    int2 warp_thread_coords;
    /// A tile loader
    block_loader_a_t loader_a;

    /// B tile loader
    block_loader_b_t loader_b;

    /// Thread's active-k/prefetch-k slices from shared A tile
    fragment_a_t local_slices_a[2][WmmaBlocksY];

    /// Thread's active-k/prefetch-k slices from shared B tile
    fragment_b_t local_slices_b[2][WmmaBlocksX];

    /// Accumulator tile
    accumulator_t accumulators[7];

    /// Thread's active-k/prefetch-k slices from shared A tile
    lds_vector_a_t local_slices_as[2][ThreadLdsVectorsA];

    /// Thread's active-k/prefetch-k slices from shared B tile
    lds_vector_b_t local_slices_bs[2][ThreadLdsVectorsB];

    /// Thread's base item offset within strip of A tile
    int thread_strip_offset_a;

    /// Thread's base item offset within strip of B tile
    int thread_strip_offset_b;

    thread_accumulator_t thread_accumulator;

    //-------------------------------------------------------------------------
    // Coordinate system helpers
    //-------------------------------------------------------------------------

    /// Compute the warp's item-coordinates (x, y) in thread block
    inline __device__
    int2 warp_item_coords()
    {
        int warp_id = threadIdx.x / WarpThreads;

        return make_int2(
            (warp_id / BlockWarpsY) * WarpItemsX,
            (warp_id % BlockWarpsY) * WarpItemsY);
    }

    // in fp16 calculation, the c slice data position
    inline __device__
    int2 thread_coords()
    {
        int lane_id = threadIdx.x % WarpThreads;

        // Maxwell+ mapping of threads within a 2D warp for maximal LDS bandwidth
        return make_int2(
            lane_id / WarpThreadsY,
            lane_id % WarpThreadsY);
    }

    /// Compute the thread block's base item-coordinates in matrix A
    inline __device__
    int2 a_block_item_coords()
    {
        if (TransformA == matrix_transform_t::NonTranspose)
        {
            return make_int2(grid_raster.block_item_coords.y, block_item_coords_k);
        }
        else
        {
            return make_int2(block_item_coords_k, grid_raster.block_item_coords.y);
        }
    }

    /// Compute the thread block's base item-coordinates in matrix B
    inline __device__
    int2 b_block_item_coords()
    {
        if (TransformB == matrix_transform_t::Transpose)
        {
            return make_int2(grid_raster.block_item_coords.x, block_item_coords_k);
        }
        else
        {
            return make_int2(block_item_coords_k, grid_raster.block_item_coords.x);
        }
    }

    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
    inline __device__
    block_task_wmma(
        scratch_storage_t *scratch,
        value_t *d_a,
        value_t *d_b,
        accum_t *d_c,
        epilogue_op_t epilogue_op,
        int dim_m,
        int dim_n,
        int dim_k,
        k_split_control k_split)
    :
        scratch(scratch),
        page_idx(0),
        d_c(d_c),
        epilogue_op(epilogue_op),
        dim_m(dim_m),
        dim_n(dim_n),
        k_split(k_split),
        block_item_coords_k(k_split.block_begin_item_k()),
        block_end_item_k(k_split.block_end_item_k(dim_k)),
        block_warp_item_coords(warp_item_coords()),
        warp_thread_coords(thread_coords()),
        loader_a(
            reinterpret_cast<value_a_t const *>(d_a),
            (IsLayoutCongruousA ? dim_m : block_end_item_k),
            (IsLayoutCongruousA ? 0 : block_item_coords_k),
            (IsLayoutCongruousA ? block_end_item_k : dim_m),
            (IsLayoutCongruousA ? dim_m : dim_k),
            (IsLayoutCongruousA ? block_item_coords_k : 0),
            a_block_item_coords()),

        loader_b(
            reinterpret_cast<value_b_t const *>(d_b),
            (IsLayoutCongruousB ? dim_n : block_end_item_k),
            (IsLayoutCongruousB ? 0 : block_item_coords_k),
            (IsLayoutCongruousB ? block_end_item_k : dim_n),
            (IsLayoutCongruousB ? dim_n : dim_k),
            (IsLayoutCongruousB ? block_item_coords_k : 0),
            b_block_item_coords()),
        thread_strip_offset_a(warp_thread_coords.y + (block_warp_item_coords.y * WarpItemsY)),
        thread_strip_offset_b(warp_thread_coords.x + (block_warp_item_coords.x * WarpItemsX))
    {}
    /**
     * Request the calling thread's slices of the shared tiles at depth \p tile_offset_k
     */
    inline __device__ void request_local_prefetch(
        fragment_a_t local_slices_a[WmmaBlocksY],  ///< Slice from A
        fragment_b_t local_slices_b[WmmaBlocksX],  ///< Slice from B
        int tile_offset_k)
    {
        value_b_t const *smem_A_base = &scratch->pages[page_idx].alias().block_a[0][0];
        value_b_t const *smem_B_base = &scratch->pages[page_idx].alias().block_b[0][0];

        int constexpr kstride_a = (IsLayoutCongruousA ? LdmSmemA : 1);
        int constexpr lstride_a = (IsLayoutCongruousA ? 1 : LdmSmemA);

        int constexpr kstride_b = (IsLayoutCongruousB ? LdmSmemB : 1);
        int constexpr lstride_b = (IsLayoutCongruousB ? 1 : LdmSmemB);

        // Load B strip
        #pragma unroll
        for (int i = 0; i < WmmaBlocksX; ++i)
        {
            value_b_t const *smem_B_ptr =
                &smem_B_base[tile_offset_k * kstride_b + (block_warp_item_coords.x + WmmaItemsX * i) * lstride_b];

            nvcuda::wmma::load_matrix_sync(local_slices_b[i], smem_B_ptr, LdmSmemB);
        }

        // Load A strip
        #pragma unroll
        for (int i = 0; i < WmmaBlocksY; ++i)
        {
            value_a_t const *smem_A_ptr =
                &smem_A_base[tile_offset_k * kstride_a + (block_warp_item_coords.y + WmmaItemsY * i) * lstride_a];

            nvcuda::wmma::load_matrix_sync(local_slices_a[i], smem_A_ptr, LdmSmemA);
        }
    }
    /**
     * Request the calling thread's slices of the shared tiles at depth \p tile_offset_k 
     */
    inline __device__ void request_local_prefetch_thread(
        lds_vector_a_t (&slice_a)[ThreadLdsVectorsA],  ///< Slice from A
        lds_vector_b_t (&slice_b)[ThreadLdsVectorsB],  ///< Slice from B
        int tile_offset_k)
    {
        // Load B strip
        /*
        
        #pragma unroll
        for (int i = 0; i < ThreadLdsVectorsB; ++i)
        {
            #pragma unroll
            for(int j = 0; j < 4; ++j){
                slice_b[i].buff[j]=scratch->pages[page_idx].alias().block_b[thread_strip_offset_b + (i * WarpThreadsX * LdsVectorDpVectorsB)][tile_offset_k];
                
            }
        }
        
        

        // Load A strip
        

        #pragma unroll
        for (int i = 0; i < ThreadLdsVectorsA; ++i)
        {
            slice_a[i].load(&scratch->pages[page_idx].alias().block_a[tile_offset_k][thread_strip_offset_a + (i * WarpThreadsY * LdsVectorDpVectorsA)]);
        }
        */
    }
    inline __device__
    void accumulator_init()
    {
        #pragma unroll
        for (int x = 0; x < 7; ++x)
        {
            nvcuda::wmma::fill_fragment(accumulators[x], accum_t(0));
        }

    }
    inline __device__
    void thread_prefetch_calc(int i)
    {
        typedef dp_vector_t thread_tile_a_t[ThreadLdsVectorsA * LdsVectorDpVectorsA];
        typedef dp_vector_t thread_tile_b_t[ThreadLdsVectorsB * LdsVectorDpVectorsB];

        //request_local_prefetch_thread(local_slices_as[(i+1)%2],local_slices_bs[(i+1)%2],(i) % BlockItemsK);
        thread_tile_a_t &thread_tile_a = reinterpret_cast<thread_tile_a_t&>(local_slices_as[i % 2]);
        thread_tile_b_t &thread_tile_b = reinterpret_cast<thread_tile_b_t&>(local_slices_bs[i % 2]);
        thread_accumulator.multiply_accumulate(thread_tile_a, thread_tile_b);
    }
    /**
     * \brief Compute the product of tile_a and tile_b and add the result to
     * the tile of accumulators.
     */
    inline __device__
    void multiply_accumulate(
        fragment_a_t (&tile_a)[WmmaBlocksY],
        fragment_b_t (&tile_b)[WmmaBlocksX],
        int tile_offset_k)
    {
        int i = 0;

        thread_prefetch_calc(tile_offset_k+i);
        thread_prefetch_calc(tile_offset_k+i+1);
        #pragma unroll
        for (int x = 0; x < WmmaBlocksX; ++x)
        {
            #pragma unroll
            for (int y = 0; y < WmmaBlocksY; ++y)
            {
                if(x==WmmaBlocksX-1&&y==WmmaBlocksY-1)
                    break;
                i+=2;
                nvcuda::wmma::mma_sync(accumulators[x*WmmaBlocksY+y], tile_a[y], tile_b[x], accumulators[x*WmmaBlocksY+y]);
                
                thread_prefetch_calc(tile_offset_k+i);
                thread_prefetch_calc(tile_offset_k+i+1);
            }
        }
    }
    //-------------------------------------------------------------------------
    // Prefetching utility methods
    //-------------------------------------------------------------------------

    

    //-------------------------------------------------------------------------
    // Epilogue
    //-------------------------------------------------------------------------

    /**
     * Performs the GEMM epilogue:
     *   - Applies the scalar multipliers and addends to the accumulators
     *   - Write the result to the output matrix
     */
    inline __device__ void epilogue()
    {
        // Wait for predecessor thread block(s) to produce partial-sums
        k_split.wait();

        // Configure epilogue as to whether the thread block is a secondary
        // accumulator in an inter-block k-splitting scheme
        if (k_split.is_secondary_accumulator())
            epilogue_op.set_secondary_accumulator();

        // Whether or not the addend from C needs loading
        bool must_init_addend = epilogue_op.must_init_addend();

        int warp_base_x = grid_raster.block_item_coords.x + block_warp_item_coords.x;
        int warp_base_y = grid_raster.block_item_coords.y + block_warp_item_coords.y;

        int constexpr SmemStride = WmmaItemsY;

        int warp_id = threadIdx.x / 32;

        // Compute shape of one accumulator read/modify/write operation
        int constexpr ItemsY = (WmmaItemsY);
        int constexpr ItemsX = (32 / ItemsY);
        int constexpr IterationsX = WmmaItemsX / ItemsX;

        // Compute a rasterization of warp lanes across the WMMA tile.
        int lane_id = (threadIdx.x % 32);
        int lane_read_x = (lane_id / ItemsY);
        int lane_read_y = (lane_id % ItemsY);

        accum_t *smem_scratch = scratch->epilogue[warp_id];
        accum_t const *smem_read_ptr = smem_scratch + lane_read_y + lane_read_x * SmemStride;

        #pragma unroll
        for (int xb = 0; xb < WmmaBlocksX; ++xb)
        {
            #pragma unroll
            for (int yb = 0; yb < WmmaBlocksY; ++yb)
            {
                if(xb==WmmaBlocksX-1&&yb==WmmaBlocksY-1)
                    break;
                // Store accumulator tile to SMEM
                nvcuda::wmma::store_matrix_sync(
                    smem_scratch,
                    accumulators[xb*WmmaBlocksY+yb],
                    SmemStride,
                    matrix_layout<matrix_transform_t::NonTranspose>::kind);

                // Synchronize threads within the warp
                __syncthreads();

                // Compute lane coordinates so that each thread efficiently accesses SMEM.
                int c_x = (warp_base_x + (xb) * WmmaItemsX + lane_read_x);
                int c_y = (warp_base_y + (yb) * WmmaItemsY + lane_read_y);

                // Compute guard predicate by comparing against problem dimensions.
                bool pred = c_y < dim_m;

                // Compute output pointer from lane coordinates
                int c_index = c_x * dim_m + c_y;
                accum_t *c_ptr = reinterpret_cast<accum_t *>(d_c) + c_x * dim_m + c_y;

                // Iterate over columns of output tile. Load from SMEM, compute epilogue operation,
                // and stream output to global memory
                #pragma unroll
                for (int item_x = 0; item_x < IterationsX; ++item_x)
                {
                    accum_t accum = smem_read_ptr[item_x * ItemsX * SmemStride];
                    accum_t c_element = 0;

                    // Filter against problem dimensions as the warp iterates across the columns of
                    // output.
                    pred = (pred && ((c_x + item_x * ItemsX) < dim_n));

                    if (must_init_addend && pred)
                    {
                        // NB: inline PTX to utilize strong operations for inter-block synchronization.
                        //     The following is equivalent to:
                        //
                                 c_element = c_ptr[0];
                        //asm volatile ("ld.global.cg.f32 %0, [%1];\n" : "=f"(c_element) : "l"(c_ptr));
                    }
                    
                    c_element = epilogue_op(accum, c_element, c_index);

                    if (pred)
                    {
                        // NB: inline PTX to utilize strong operations for inter-block synchronization.
                        //     The following is equivalent to:
                        //
                                 c_ptr[0] = c_element;

                        //asm volatile ("st.global.cg.f32 [%0], %1;\n" : : "l"(c_ptr), "f"(c_element));
                    }

                    // Increment output pointer
                    c_ptr += dim_m * ItemsX;
                    c_index += dim_m * ItemsX;
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for(int xb=0;xb<ThreadItemsX;xb++){
            #pragma unroll
            for(int yb=0;yb<ThreadItemsY;yb++){
                accum_t accum=thread_accumulator.get(xb,yb);
                int c_x = (warp_base_x + 3 * WmmaItemsX + lane_read_x);
                int c_y = (warp_base_y + 2 * WmmaItemsY + lane_read_y);
                int c_index = (c_x+xb) * dim_m + c_y+yb;
                accum_t *c_ptr = reinterpret_cast<accum_t *>(d_c) + (c_x+xb) * dim_m + c_y+yb;
                accum_t c_element = c_ptr[0];
                c_element = epilogue_op(accum, c_element, c_index);
                c_ptr[0] = c_element;
            }
        }

        // Signal k-split successor thread_block
        k_split.signal();
    }

    //-------------------------------------------------------------------------
    // Tile consumption
    //-------------------------------------------------------------------------

    /**
     * Consume a tile of A and B each
     */
    template <bool DoGlobalPrefetch>
    inline __device__
    void consume_tile()
    {
        // Request global prefetch for next tile on first strip
        if (DoGlobalPrefetch)
        {
            loader_b.request();
            loader_b.next();
            loader_a.request();
            loader_a.next();
        }

        // Unroll BlockDpVectorsK iterations of outer-product accumulations
        #pragma unroll
        for (int iteration = 0; iteration < WmmaUnrollCount; ++iteration)
        {
            int tile_offset_k = iteration * WmmaItemsK;

            // Active load-from-shared index
            int active_lds_idx = __NV_STD_MIN(WmmaUnrollCount - 1, (iteration) % 2);

            // Next load-from-shared index
            int next_lds_idx = __NV_STD_MIN(WmmaUnrollCount - 1, (iteration + 1) % 2);

            // The last unrolled iteration commits the global fetches
            if ((iteration == WmmaUnrollCount - 1) && DoGlobalPrefetch)
            {
                // If not using two pages of scratch tiles, protect the above prefetch loads from
                // the committing writes below
                if (!UseDoubleScratchTiles)
                {
                    __syncthreads();
                }
                else
                {
                    page_idx = (page_idx ? 0 : 1);
                }

                // Commit global prefetch data to scratch page
                loader_a.template commit<LdmSmemA>(&scratch->pages[page_idx].alias().block_a[0][0]);
                loader_b.template commit<LdmSmemB>(&scratch->pages[page_idx].alias().block_b[0][0]);

                __syncthreads();
            }

            // Accumulate this dp-stripe product
            multiply_accumulate(
                local_slices_a[active_lds_idx],
                local_slices_b[active_lds_idx],
                tile_offset_k);

            // Request local prefetch for next strip
            request_local_prefetch(
                local_slices_a[next_lds_idx],
                local_slices_b[next_lds_idx],
                (tile_offset_k + WmmaItemsK) % BlockItemsK);
            /*
            #pragma unroll
            for(int i=0;i<16;i++){
                typedef dp_vector_t thread_tile_a_t[ThreadLdsVectorsA * LdsVectorDpVectorsA];
                typedef dp_vector_t thread_tile_b_t[ThreadLdsVectorsB * LdsVectorDpVectorsB];

                //request_local_prefetch_thread(local_slices_as[(i+1)%2],local_slices_bs[(i+1)%2],(i) % BlockItemsK);
                thread_tile_a_t &thread_tile_a = reinterpret_cast<thread_tile_a_t&>(local_slices_as[i % 2]);
                thread_tile_b_t &thread_tile_b = reinterpret_cast<thread_tile_b_t&>(local_slices_bs[i % 2]);
                thread_accumulator.multiply_accumulate(thread_tile_a, thread_tile_b);
            }
            */
                //thread_prefetch_calc(tile_offset_k+i);
        }
            
    }

    //-------------------------------------------------------------------------
    // GEMM API
    //-------------------------------------------------------------------------

    /**
     * Compute GEMM
     */
    inline __device__
    void run()
    {
        // Quit if the thread block is fully out-of-bounds
        if (grid_raster.is_block_oob(dim_m, dim_n))
        {
            asm volatile("exit;");
        }

        // Request global prefetch of first tile
        loader_a.request();
        loader_a.next();
        loader_b.request();
        loader_b.next();

        // Commit global prefetch of first tile to shared memory
        loader_a.template commit<LdmSmemA>(&scratch->pages[page_idx].alias().block_a[0][0]);
        loader_b.template commit<LdmSmemB>(&scratch->pages[page_idx].alias().block_b[0][0]);

        // Advance to next A,B tiles in K-axis
        block_item_coords_k += BlockItemsK;

        // Synchronize shared tiles and prepared accumulator
        __syncthreads();

        // Initialize thread's slice of accumulators
        accumulator_init();
        // Request first iteration of local prefetch strips
        request_local_prefetch(
            local_slices_a[0],
            local_slices_b[0],
            0);

        //
        // Main loop
        //

        // Consume tiles in A and B along the K-axis (all but last tile)
        #pragma unroll 1
        while (block_item_coords_k < block_end_item_k)
        {
            consume_tile<true>();

            // Advance to next A,B tiles in K-axis
            block_item_coords_k += BlockItemsK;
        }

        consume_tile<false>();

        //
        // Eplilogue
        //

        // prevent overwriting SMEM until all warps have finished loading data
        __syncthreads();

        // store accumulator tile to global memory
        epilogue();
    }
};

} // namespace gemm
} // namespace cutlass

#endif
