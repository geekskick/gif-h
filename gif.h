//
// gif.h
// by Charlie Tangora
// Public domain.
// Email me : ctangora -at- gmail -dot- com
//
// This file offers a simple, very limited way to create animated GIFs directly in code.
//
// Those looking for particular cleverness are likely to be disappointed; it's pretty
// much a straight-ahead implementation of the GIF format with optional Floyd-Steinberg
// dithering. (It does at least use delta encoding - only the changed portions of each
// frame are saved.)
//
// So resulting files are often quite large. The hope is that it will be handy nonetheless
// as a quick and easily-integrated way for programs to spit out animations.
//
// Only RGBA8 is currently supported as an input format. (The alpha is ignored.)
//
// USAGE:
// Create a GifWriter struct. Pass it to GifBegin() to initialize and write the header.
// Pass subsequent frames to GifWriteFrame().
// Finally, call GifEnd() to close the file handle and free memory.
//

#ifndef gif_h
#define gif_h

#include <stdio.h>   // for FILE*
#include <string.h>  // for memcpy and bzero
#include <stdint.h>  // for integer typedefs

// Define these macros to hook into a custom memory allocator.
// TEMP_MALLOC and TEMP_FREE will only be called in stack fashion - frees in the reverse order of mallocs
// and any temp memory allocated by a function will be freed before it exits.
// MALLOC and FREE are used only by GifBegin and GifEnd respectively (to allocate a buffer the size of the image, which
// is used to find changed pixels for delta-encoding.)

#ifndef GIF_TEMP_MALLOC
#include <stdlib.h>
#define GIF_TEMP_MALLOC malloc
#endif

#ifndef GIF_TEMP_FREE
#include <stdlib.h>
#define GIF_TEMP_FREE free
#endif

#ifndef GIF_MALLOC
#include <stdlib.h>
#define GIF_MALLOC malloc
#endif

#ifndef GIF_FREE
#include <stdlib.h>
#define GIF_FREE free
#endif

const int gif_transparent_idx = 0;

struct gif_palette
{
    int bit_depth;

    uint8_t r[256];
    uint8_t g[256];
    uint8_t b[256];

    // k-d tree over RGB space, organized in heap fashion
    // i.e. left child of node i is node i*2, right child is node i*2+1
    // nodes 256-511 are implicitly the leaves, containing a color
    uint8_t tree_split_elt[255];
    uint8_t tree_split[255];
};

// max, min, and abs functions
int gif_i_max(int l, int r) { return l>r?l:r; }
int gif_i_min(int l, int r) { return l<r?l:r; }
int gif_i_abs(int i) { return i<0?-i:i; }

// walks the k-d tree to pick the palette entry for a desired color.
// Takes as in/out parameters the current best color and its error -
// only changes them if it finds a better color in its subtree.
// this is the major hotspot in the code at the moment.
void git_get_closest_palette_colour(struct gif_palette* p_pal, int r, int g, int b, int* best_ind, int* best_diff, int tree_root)
{
    // base case, reached the bottom of the tree
    if(tree_root > (1<<p_pal->bit_depth)-1)
    {
        int ind = tree_root-(1<<p_pal->bit_depth);
        if(ind == gif_transparent_idx) return;

        // check whether this color is better than the current winner
        int r_err = r - ((int32_t)p_pal->r[ind]);
        int g_err = g - ((int32_t)p_pal->g[ind]);
        int b_err = b - ((int32_t)p_pal->b[ind]);
        int diff = gif_i_abs(r_err)+gif_i_abs(g_err)+gif_i_abs(b_err);

        if(diff < *best_diff)
        {
            *best_ind = ind;
            *best_diff = diff;
        }

        return;
    }

    // take the appropriate color (r, g, or b) for this node of the k-d tree
    int comps[3]; comps[0] = r; comps[1] = g; comps[2] = b;
    int split_comp = comps[p_pal->tree_split_elt[tree_root]];

    int split_pos = p_pal->tree_split[tree_root];
    if(split_pos > split_comp)
    {
        // check the left subtree
        git_get_closest_palette_colour(p_pal, r, g, b, best_ind, best_diff, tree_root*2);
        if( *best_diff > split_pos - split_comp )
        {
            // cannot prove there's not a better value in the right subtree, check that too
            git_get_closest_palette_colour(p_pal, r, g, b, best_ind, best_diff, tree_root*2+1);
        }
    }
    else
    {
        git_get_closest_palette_colour(p_pal, r, g, b, best_ind, best_diff, tree_root*2+1);
        if( *best_diff > split_comp - split_pos )
        {
            git_get_closest_palette_colour(p_pal, r, g, b, best_ind, best_diff, tree_root*2);
        }
    }
}

void gif_swap_pixels(uint8_t* image, int pix_a, int pix_b)
{
    uint8_t rA = image[pix_a*4];
    uint8_t gA = image[pix_a*4+1];
    uint8_t bA = image[pix_a*4+2];
    uint8_t aA = image[pix_a*4+3];

    uint8_t rB = image[pix_b*4];
    uint8_t gB = image[pix_b*4+1];
    uint8_t bB = image[pix_b*4+2];
    uint8_t aB = image[pix_a*4+3];

    image[pix_a*4] = rB;
    image[pix_a*4+1] = gB;
    image[pix_a*4+2] = bB;
    image[pix_a*4+3] = aB;

    image[pix_b*4] = rA;
    image[pix_b*4+1] = gA;
    image[pix_b*4+2] = bA;
    image[pix_b*4+3] = aA;
}

// just the partition operation from quicksort
int gif_partition(uint8_t* image, const int left, const int right, const int elt, int pivot_idx)
{
    const int pivot_value = image[(pivot_idx)*4+elt];
    gif_swap_pixels(image, pivot_idx, right-1);
    int store_index = left;
    bool split = 0;
    for(int ii=left; ii<right-1; ++ii)
    {
        int array_val = image[ii*4+elt];
        if( array_val < pivot_value )
        {
            gif_swap_pixels(image, ii, store_index);
            ++store_index;
        }
        else if( array_val == pivot_value )
        {
            if(split)
            {
                gif_swap_pixels(image, ii, store_index);
                ++store_index;
            }
            split = !split;
        }
    }
    gif_swap_pixels(image, store_index, right-1);
    return store_index;
}

// Perform an incomplete sort, finding all elements above and below the desired median
void gif_partition_by_median(uint8_t* image, int left, int right, int com, int needed_centre)
{
    if(left < right-1)
    {
        int pivot_idx = left + (right-left)/2;

        pivot_idx = gif_partition(image, left, right, com, pivot_idx);

        // Only "sort" the section of the array that contains the median
        if(pivot_idx > needed_centre)
            gif_partition_by_median(image, left, pivot_idx, com, needed_centre);

        if(pivot_idx < needed_centre)
            gif_partition_by_median(image, pivot_idx+1, right, com, needed_centre);
    }
}

// Builds a palette by creating a balanced k-d tree of all pixels in the image
void gif_split_palette(uint8_t* image, int num_pixels, int first_elt, int last_elt, int split_elt, int split_dist, int tree_node, bool build_for_dither, struct gif_palette* p_pal)
{
    if(last_elt <= first_elt || num_pixels == 0)
        return;

    // base case, bottom of the tree
    if(last_elt == first_elt+1)
    {
        if(build_for_dither)
        {
            // Dithering needs at least one color as dark as anything
            // in the image and at least one brightest color -
            // otherwise it builds up error and produces strange artifacts
            if( first_elt == 1 )
            {
                // special case: the darkest color in the image
                uint32_t r=255, g=255, b=255;
                for(int ii=0; ii<num_pixels; ++ii)
                {
                    r = (uint32_t)gif_i_min((int32_t)r, image[ii * 4 + 0]);
                    g = (uint32_t)gif_i_min((int32_t)g, image[ii * 4 + 1]);
                    b = (uint32_t)gif_i_min((int32_t)b, image[ii * 4 + 2]);
                }

                p_pal->r[first_elt] = (uint8_t)r;
                p_pal->g[first_elt] = (uint8_t)g;
                p_pal->b[first_elt] = (uint8_t)b;

                return;
            }

            if( first_elt == (1 << p_pal->bit_depth)-1 )
            {
                // special case: the lightest color in the image
                uint32_t r=0, g=0, b=0;
                for(int ii=0; ii<num_pixels; ++ii)
                {
                    r = (uint32_t)gif_i_max((int32_t)r, image[ii * 4 + 0]);
                    g = (uint32_t)gif_i_max((int32_t)g, image[ii * 4 + 1]);
                    b = (uint32_t)gif_i_max((int32_t)b, image[ii * 4 + 2]);
                }

                p_pal->r[first_elt] = (uint8_t)r;
                p_pal->g[first_elt] = (uint8_t)g;
                p_pal->b[first_elt] = (uint8_t)b;

                return;
            }
        }

        // otherwise, take the average of all colors in this subcube
        uint64_t r=0, g=0, b=0;
        for(int ii=0; ii<num_pixels; ++ii)
        {
            r += image[ii*4+0];
            g += image[ii*4+1];
            b += image[ii*4+2];
        }

        r += (uint64_t)num_pixels / 2;  // round to nearest
        g += (uint64_t)num_pixels / 2;
        b += (uint64_t)num_pixels / 2;

        r /= (uint64_t)num_pixels;
        g /= (uint64_t)num_pixels;
        b /= (uint64_t)num_pixels;

        p_pal->r[first_elt] = (uint8_t)r;
        p_pal->g[first_elt] = (uint8_t)g;
        p_pal->b[first_elt] = (uint8_t)b;

        return;
    }

    // Find the axis with the largest range
    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;
    for(int ii=0; ii<num_pixels; ++ii)
    {
        int r = image[ii*4+0];
        int g = image[ii*4+1];
        int b = image[ii*4+2];

        if(r > maxR) maxR = r;
        if(r < minR) minR = r;

        if(g > maxG) maxG = g;
        if(g < minG) minG = g;

        if(b > maxB) maxB = b;
        if(b < minB) minB = b;
    }

    int rRange = maxR - minR;
    int gRange = maxG - minG;
    int bRange = maxB - minB;

    // and split along that axis. (incidentally, this means this isn't a "proper" k-d tree but I don't know what else to call it)
    int split_com = 1;
    if(bRange > gRange) split_com = 2;
    if(rRange > bRange && rRange > gRange) split_com = 0;

    int sub_pixels_a = num_pixels * (split_elt - first_elt) / (last_elt - first_elt);
    int sub_pixels_b = num_pixels-sub_pixels_a;

    gif_partition_by_median(image, 0, num_pixels, split_com, sub_pixels_a);

    p_pal->tree_split_elt[tree_node] = (uint8_t)split_com;
    p_pal->tree_split[tree_node] = image[sub_pixels_a*4+split_com];

    gif_split_palette(image,              sub_pixels_a, first_elt, split_elt, split_elt-split_dist, split_dist/2, tree_node*2,   build_for_dither, p_pal);
    gif_split_palette(image+sub_pixels_a*4, sub_pixels_b, split_elt, last_elt,  split_elt+split_dist, split_dist/2, tree_node*2+1, build_for_dither, p_pal);
}

// Finds all pixels that have changed from the previous image and
// moves them to the fromt of th buffer.
// This allows us to build a palette optimized for the colors of the
// changed pixels only.
int gif_pick_changed_pixels( const uint8_t* last_frame, uint8_t* frame, int num_pixels )
{
    int num_changed = 0;
    uint8_t* write_iter = frame;

    for (int ii=0; ii<num_pixels; ++ii)
    {
        if(last_frame[0] != frame[0] ||
           last_frame[1] != frame[1] ||
           last_frame[2] != frame[2])
        {
            write_iter[0] = frame[0];
            write_iter[1] = frame[1];
            write_iter[2] = frame[2];
            ++num_changed;
            write_iter += 4;
        }
        last_frame += 4;
        frame += 4;
    }

    return num_changed;
}

// The size of a gif image frame is the width*height, and as each pixel is represented as an
// RGBA colour it's effectively w*h*4.
size_t gif_get_size(const int width, const int height){
	return (size_t)(width * height * sizeof(uint32_t));
}
// Creates a palette by placing all the image pixels in a k-d tree and then averaging the blocks at the bottom.
// This is known as the "modified median split" technique
void gif_make_palette( const uint8_t* last_frame, const uint8_t* next_frame, uint32_t width, uint32_t height, int bit_depth, bool dither, struct gif_palette* p_pal )
{
    p_pal->bit_depth = bit_depth;
	const size_t image_size = gif_get_size(width, height);
	
    // SplitPalette is destructive (it sorts the pixels by color) so
    // we must create a copy of the image for it to destroy
    uint8_t* destroyable_image = (uint8_t*)GIF_TEMP_MALLOC(image_size);
	if(NULL == destroyable_image){
		fprintf(stderr, "[%s:%d]\tUnable to allocate memory for destroyable_image\n", __FILE__, __LINE__);
		return;
	}
    memcpy(destroyable_image, next_frame, image_size);

    int num_pixels = (int)(width * height);
    if(last_frame)
        num_pixels = gif_pick_changed_pixels(last_frame, destroyable_image, num_pixels);

    const int last_elt = 1 << bit_depth;
    const int split_elt = last_elt/2;
    const int split_dist = split_elt/2;

    gif_split_palette(destroyable_image, num_pixels, 1, last_elt, split_elt, split_dist, 1, dither, p_pal);

    GIF_TEMP_FREE(destroyable_image);

    // add the bottom node for the transparency index
    p_pal->tree_split[1 << (bit_depth-1)] = 0;
    p_pal->tree_split_elt[1 << (bit_depth-1)] = 0;

    p_pal->r[0] = p_pal->g[0] = p_pal->b[0] = 0;
}

// Implements Floyd-Steinberg dithering, writes palette value to alpha
void gif_dither_image( const uint8_t* last_frame, const uint8_t* next_frame, uint8_t* out_frame, uint32_t width, uint32_t height, struct gif_palette* p_pal )
{
    int num_pixels = (int)(width * height);

    // quantPixels initially holds color*256 for all pixels
    // The extra 8 bits of precision allow for sub-single-color error values
    // to be propagated
    int32_t *quant_pixels = (int32_t *)GIF_TEMP_MALLOC(sizeof(int32_t) * (size_t)num_pixels * 4);

    for( int ii=0; ii<num_pixels*4; ++ii )
    {
        uint8_t pix = next_frame[ii];
        int32_t pix16 = ((int32_t)pix) * 256;
        quant_pixels[ii] = pix16;
    }

    for( uint32_t yy=0; yy<height; ++yy )
    {
        for( uint32_t xx=0; xx<width; ++xx )
        {
            int32_t* next_pix = quant_pixels + 4*(yy*width+xx);
            const uint8_t* last_pix = last_frame? last_frame + 4*(yy*width+xx) : NULL;

            // Compute the colors we want (rounding to nearest)
            int32_t rr = (next_pix[0] + 127) / 256;
            int32_t gg = (next_pix[1] + 127) / 256;
            int32_t bb = (next_pix[2] + 127) / 256;

            // if it happens that we want the color from last frame, then just write out
            // a transparent pixel
            if( last_frame &&
               last_pix[0] == rr &&
               last_pix[1] == gg &&
               last_pix[2] == bb )
            {
                next_pix[0] = rr;
                next_pix[1] = gg;
                next_pix[2] = bb;
                next_pix[3] = gif_transparent_idx;
                continue;
            }

            int32_t best_diff = 1000000;
            int32_t best_idx = gif_transparent_idx;

            // Search the palete
            git_get_closest_palette_colour(p_pal, rr, gg, bb, &best_idx, &best_diff, 1);

            // Write the result to the temp buffer
            int32_t r_err = next_pix[0] - (int32_t)p_pal->r[best_idx] * 256;
            int32_t g_err = next_pix[1] - (int32_t)p_pal->g[best_idx] * 256;
            int32_t b_err = next_pix[2] - (int32_t)p_pal->b[best_idx] * 256;

            next_pix[0] = p_pal->r[best_idx];
            next_pix[1] = p_pal->g[best_idx];
            next_pix[2] = p_pal->b[best_idx];
            next_pix[3] = best_idx;

            // Propagate the error to the four adjacent locations
            // that we haven't touched yet
            int quantloc_7 = (int)(yy * width + xx + 1);
            int quantloc_3 = (int)(yy * width + width + xx - 1);
            int quantloc_5 = (int)(yy * width + width + xx);
            int quantloc_1 = (int)(yy * width + width + xx + 1);

            if(quantloc_7 < num_pixels)
            {
                int32_t* pix7 = quant_pixels+4*quantloc_7;
                pix7[0] += gif_i_max( -pix7[0], r_err * 7 / 16 );
                pix7[1] += gif_i_max( -pix7[1], g_err * 7 / 16 );
                pix7[2] += gif_i_max( -pix7[2], b_err * 7 / 16 );
            }

            if(quantloc_3 < num_pixels)
            {
                int32_t* pix3 = quant_pixels+4*quantloc_3;
                pix3[0] += gif_i_max( -pix3[0], r_err * 3 / 16 );
                pix3[1] += gif_i_max( -pix3[1], g_err * 3 / 16 );
                pix3[2] += gif_i_max( -pix3[2], b_err * 3 / 16 );
            }

            if(quantloc_5 < num_pixels)
            {
                int32_t* pix5 = quant_pixels+4*quantloc_5;
                pix5[0] += gif_i_max( -pix5[0], r_err * 5 / 16 );
                pix5[1] += gif_i_max( -pix5[1], g_err * 5 / 16 );
                pix5[2] += gif_i_max( -pix5[2], b_err * 5 / 16 );
            }

            if(quantloc_1 < num_pixels)
            {
                int32_t* pix1 = quant_pixels+4*quantloc_1;
                pix1[0] += gif_i_max( -pix1[0], r_err / 16 );
                pix1[1] += gif_i_max( -pix1[1], g_err / 16 );
                pix1[2] += gif_i_max( -pix1[2], b_err / 16 );
            }
        }
    }

    // Copy the palettized result to the output buffer
    for( int ii=0; ii<num_pixels*4; ++ii )
    {
        out_frame[ii] = (uint8_t)quant_pixels[ii];
    }

    GIF_TEMP_FREE(quant_pixels);
}

// Picks palette colors for the image using simple thresholding, no dithering
void gif_threshold_image( const uint8_t* last_frame, const uint8_t* next_frame, uint8_t* out_frame, uint32_t width, uint32_t height, struct gif_palette* p_pal )
{
    uint32_t num_pixels = width * height;
    for( uint32_t ii=0; ii<num_pixels; ++ii )
    {
        // if a previous color is available, and it matches the current color,
        // set the pixel to transparent
        if(last_frame &&
           last_frame[0] == next_frame[0] &&
           last_frame[1] == next_frame[1] &&
           last_frame[2] == next_frame[2])
        {
            out_frame[0] = last_frame[0];
            out_frame[1] = last_frame[1];
            out_frame[2] = last_frame[2];
            out_frame[3] = gif_transparent_idx;
        }
        else
        {
            // palettize the pixel
            int32_t best_diff = 1000000;
            int32_t best_idx = 1;
            git_get_closest_palette_colour(p_pal, next_frame[0], next_frame[1], next_frame[2], &best_idx, &best_diff, 1);

            // Write the resulting color to the output buffer
            out_frame[0] = p_pal->r[best_idx];
            out_frame[1] = p_pal->g[best_idx];
            out_frame[2] = p_pal->b[best_idx];
            out_frame[3] = (uint8_t)best_idx;
        }

        if(last_frame) last_frame += 4;
        out_frame += 4;
        next_frame += 4;
    }
}

// Simple structure to write out the LZW-compressed portion of the image
// one bit at a time
struct gif_bit_status
{
    uint8_t bit_idx;  // how many bits in the partial byte written so far
    uint8_t byte;      // current partial byte

    uint32_t chunk_idx;
    uint8_t chunk[256];   // bytes are written in here until we have 256 of them, then written to the file
};

// insert a single bit
void gif_write_bit( struct gif_bit_status* stat, uint32_t bit )
{
    bit = bit & 1;
    bit = bit << stat->bit_idx;
    stat->byte |= bit;

    ++stat->bit_idx;
    if( stat->bit_idx > 7 )
    {
        // move the newly-finished byte to the chunk buffer
        stat->chunk[stat->chunk_idx++] = stat->byte;
        // and start a new byte
        stat->bit_idx = 0;
        stat->byte = 0;
    }
}

// write all bytes so far to the file
void gif_write_chunk( FILE* f, struct gif_bit_status* stat )
{
    fputc((int)stat->chunk_idx, f);
    fwrite(stat->chunk, 1, stat->chunk_idx, f);

    stat->bit_idx = 0;
    stat->byte = 0;
    stat->chunk_idx = 0;
}

void gif_write_code( FILE* f, struct gif_bit_status* stat, uint32_t code, uint32_t length )
{
    for( uint32_t ii=0; ii<length; ++ii )
    {
        gif_write_bit(stat, code);
        code = code >> 1;

        if( stat->chunk_idx == 255 )
        {
            gif_write_chunk(f, stat);
        }
    }
}

// The LZW dictionary is a 256-ary tree constructed as the file is encoded,
// this is one node
struct gif_lzw_code
{
    uint16_t next[256];
};

// write a 256-color (8-bit) image palette to the file
void gif_write_palette(struct gif_palette* p_pal, FILE* f )
{
    fputc(0, f);  // first color: transparency
    fputc(0, f);
    fputc(0, f);

    for(int ii=1; ii<(1 << p_pal->bit_depth); ++ii)
    {
        uint32_t r = p_pal->r[ii];
        uint32_t g = p_pal->g[ii];
        uint32_t b = p_pal->b[ii];

        fputc((int)r, f);
        fputc((int)g, f);
        fputc((int)b, f);
    }
}

// write the image header, LZW-compress and write out the image
void gif_write_lzw_image(FILE* f, uint8_t* image, uint32_t left, uint32_t top,  uint32_t width, uint32_t height, uint32_t delay, struct gif_palette* p_pal)
{
    // graphics control extension
    fputc(0x21, f);
    fputc(0xf9, f);
    fputc(0x04, f);
    fputc(0x05, f); // leave prev frame in place, this frame has transparency
    fputc(delay & 0xff, f);
    fputc((delay >> 8) & 0xff, f);
    fputc(gif_transparent_idx, f); // transparent color index
    fputc(0, f);

    fputc(0x2c, f); // image descriptor block

    fputc(left & 0xff, f);           // corner of image in canvas space
    fputc((left >> 8) & 0xff, f);
    fputc(top & 0xff, f);
    fputc((top >> 8) & 0xff, f);

    fputc(width & 0xff, f);          // width and height of image
    fputc((width >> 8) & 0xff, f);
    fputc(height & 0xff, f);
    fputc((height >> 8) & 0xff, f);

    //fputc(0, f); // no local color table, no transparency
    //fputc(0x80, f); // no local color table, but transparency

    fputc(0x80 + p_pal->bit_depth-1, f); // local color table present, 2 ^ bitDepth entries
    gif_write_palette(p_pal, f);

    const int min_code_size = p_pal->bit_depth;
    const uint32_t clear_code = 1 << p_pal->bit_depth;

    fputc(min_code_size, f); // min code size 8 bits

    struct gif_lzw_code* codetree = (struct gif_lzw_code*)GIF_TEMP_MALLOC(sizeof(struct gif_lzw_code)*4096);

    memset(codetree, 0, sizeof(struct gif_lzw_code)*4096);
    int32_t current_code = -1;
    uint32_t code_size = (uint32_t)min_code_size + 1;
    uint32_t max_code = clear_code+1;

    struct gif_bit_status stat;
    stat.byte = 0;
    stat.bit_idx = 0;
    stat.chunk_idx = 0;

    gif_write_code(f, &stat, clear_code, code_size);  // start with a fresh LZW dictionary

    for(uint32_t yy=0; yy<height; ++yy)
    {
        for(uint32_t xx=0; xx<width; ++xx)
        {
            uint8_t next_value = image[(yy*width+xx)*4+3];

            // "loser mode" - no compression, every single code is followed immediately by a clear
            //WriteCode( f, stat, nextValue, codeSize );
            //WriteCode( f, stat, 256, codeSize );

            if( current_code < 0 )
            {
                // first value in a new run
                current_code = next_value;
            }
            else if( codetree[current_code].next[next_value] )
            {
                // current run already in the dictionary
                current_code = codetree[current_code].next[next_value];
            }
            else
            {
                // finish the current run, write a code
                gif_write_code(f, &stat, (uint32_t)current_code, code_size);

                // insert the new run into the dictionary
                codetree[current_code].next[next_value] = (uint16_t)++max_code;

                if( max_code >= (1ul << code_size) )
                {
                    // dictionary entry count has broken a size barrier,
                    // we need more bits for codes
                    code_size++;
                }
                if( max_code == 4095 )
                {
                    // the dictionary is full, clear it out and begin anew
                    gif_write_code(f, &stat, clear_code, code_size); // clear tree

                    memset(codetree, 0, sizeof(struct gif_lzw_code)*4096);
                    code_size = (uint32_t)(min_code_size + 1);
                    max_code = clear_code+1;
                }

                current_code = next_value;
            }
        }
    }

    // compression footer
    gif_write_code(f, &stat, (uint32_t)current_code, code_size);
    gif_write_code(f, &stat, clear_code, code_size);
    gif_write_code(f, &stat, clear_code + 1, (uint32_t)min_code_size + 1);

    // write out the last partial chunk
    while( stat.bit_idx ) gif_write_bit(&stat, 0);
    if( stat.chunk_idx ) gif_write_chunk(f, &stat);

    fputc(0, f); // image block terminator

    GIF_TEMP_FREE(codetree);
}

struct gif_writer
{
    FILE* f;
    uint8_t* old_image;
    bool first_frame;
};

// Creates a gif file.
// The input GIFWriter is assumed to be uninitialized.
// The delay value is the time between frames in hundredths of a second - note that not all viewers pay much attention to this value.
bool GifBegin( struct gif_writer* writer, const char* filename, uint32_t width, uint32_t height, uint32_t delay, int32_t bit_depth, bool dither )
{
    (void)bit_depth; (void)dither; // Mute "Unused argument" warnings
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	writer->f = 0;
    fopen_s(&writer->f, filename, "wb");
#else
    writer->f = fopen(filename, "wb");
#endif
    if(!writer->f) return false;

    writer->first_frame = true;

    // allocate
    writer->old_image = (uint8_t*)GIF_MALLOC(width*height*4);
	if(NULL == writer->old_image){
		fprintf(stderr, "[%s:%d]\tError allocating %u bytes for writer->old_image\n", __FILE__, __LINE__, width*height*4);
		return false;
	}

    fputs("GIF89a", writer->f);

    // screen descriptor
    fputc(width & 0xff, writer->f);
    fputc((width >> 8) & 0xff, writer->f);
    fputc(height & 0xff, writer->f);
    fputc((height >> 8) & 0xff, writer->f);

    fputc(0xf0, writer->f);  // there is an unsorted global color table of 2 entries
    fputc(0, writer->f);     // background color
    fputc(0, writer->f);     // pixels are square (we need to specify this because it's 1989)

    // now the "global" palette (really just a dummy palette)
    // color 0: black
    fputc(0, writer->f);
    fputc(0, writer->f);
    fputc(0, writer->f);
    // color 1: also black
    fputc(0, writer->f);
    fputc(0, writer->f);
    fputc(0, writer->f);

    if( delay != 0 )
    {
        // animation header
        fputc(0x21, writer->f); // extension
        fputc(0xff, writer->f); // application specific
        fputc(11, writer->f); // length 11
        fputs("NETSCAPE2.0", writer->f); // yes, really
        fputc(3, writer->f); // 3 bytes of NETSCAPE2.0 data

        fputc(1, writer->f); // JUST BECAUSE
        fputc(0, writer->f); // loop infinitely (byte 0)
        fputc(0, writer->f); // loop infinitely (byte 1)

        fputc(0, writer->f); // block terminator
    }

    return true;
}

// Writes out a new frame to a GIF in progress.
// The GIFWriter should have been created by GIFBegin.
// AFAIK, it is legal to use different bit depths for different frames of an image -
// this may be handy to save bits in animations that don't change much.
bool gif_write_frame( struct gif_writer* writer, const uint8_t* image, uint32_t width, uint32_t height, uint32_t delay, int bit_depth, bool dither )
{
    if(!writer->f) return false;

    const uint8_t* old_image = writer->first_frame? NULL : writer->old_image;
    writer->first_frame = false;

    struct gif_palette pal;
    gif_make_palette((dither? NULL : old_image), image, width, height, bit_depth, dither, &pal);

    if(dither)
        gif_dither_image(old_image, image, writer->old_image, width, height, &pal);
    else
        gif_threshold_image(old_image, image, writer->old_image, width, height, &pal);

    gif_write_lzw_image(writer->f, writer->old_image, 0, 0, width, height, delay, &pal);

    return true;
}

// Writes the EOF code, closes the file handle, and frees temp memory used by a GIF.
// Many if not most viewers will still display a GIF properly if the EOF code is missing,
// but it's still a good idea to write it out.
bool gif_end( struct gif_writer* writer )
{
	if(!writer->f) {
		return false;
	}

    fputc(0x3b, writer->f); // end of file
    fclose(writer->f);
    GIF_FREE(writer->old_image);

    writer->f = NULL;
    writer->old_image = NULL;

    return true;
}

#endif
