/*
Original Author: ryanleh
Modified Work Copyright (c) 2020 Microsoft Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Modified by Deevashwer Rathee
*/

#ifndef CONV_FIELD_H__
#define CONV_FIELD_H__

#include <set>
#include "LinearHE/utils-HE.h"
#include <Eigen/Dense>

// This is to keep compatibility for im2col implementations
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Channel;
typedef std::vector<Channel> Image;
typedef std::vector<Image> Filters;

struct ConvMetadata {
    int slot_count;
    // Number of plaintext slots in a half ciphertext
    // (since ciphertexts are a two column matrix)
    int32_t pack_num;
    // Number of Channels that can fit in a half ciphertext
    int32_t chans_per_half;
    // Number of Channels that can fit in a ciphertext
    int32_t chans_per_cipher;
    // Number of input ciphertexts for convolution
    int32_t inp_ct;
    // Number of output ciphertexts
    int32_t out_ct;
    // Image and Filters metadata
    int32_t image_h;
    int32_t image_w;
    int32_t image_size;
    int32_t inp_chans;
    int32_t filter_h;
    int32_t filter_w;
    int32_t filter_size;
    int32_t out_chans;
    // How many total ciphertext halves the input and output take up
    int32_t inp_halves;
    int32_t out_halves;
    // The modulo used when deciding which output channels to pack into a mask
    int32_t out_mod;
    // How many permutations of ciphertexts are needed to generate all
    // intermediate rotation sets
    int32_t half_perms;
    /* The number of rotations for each ciphertext half */
    int32_t half_rots;
    // Total number of convolutions needed to generate all
    // intermediate rotations sets
    int32_t convs;
    
    int32_t stride_h;
    int32_t stride_w;
    int32_t output_h;
    int32_t output_w;
    int32_t pad_t;
    int32_t pad_b;
    int32_t pad_r;
    int32_t pad_l;
};

//this function performs independent ideal function for convolution
Image ideal_function(
        Image &image,
        Filters &filters,
        ConvMetadata data);
                      


/* Use casting to do two conditionals instead of one - check if a > 0 and a < b */
inline bool condition_check(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

//get the greatest common divisor of two numbers
int GCD(int a, int b);

//get the least common multiple of two numbers
inline int LCM(int a, int b){
    return (a / GCD(a, b)) * b;
}


Image pad_image(
        ConvMetadata data,
        Image &image);

void i2c(
        Image &image,
        Channel &column,
        const int filter_h,
        const int filter_w, 
        const int stride_h,
        const int stride_w,
        const int output_h,
        const int output_w);

std::vector<seal::Ciphertext> HE_preprocess_noise(
        const uint64_t* const* secret_share,
        const ConvMetadata &data,
        seal::Encryptor &encryptor,
        seal::BatchEncoder &batch_encoder,
        seal::Evaluator &evaluator);

std::vector<std::vector<uint64_t>> preprocess_image_OP(
        Image &image,
        ConvMetadata data);

std::vector<std::vector<seal::Ciphertext>> filter_rotations(
        std::vector<seal::Ciphertext> &input,
        const ConvMetadata &data,
        seal::Evaluator *evaluator = NULL,
        seal::GaloisKeys *gal_keys = NULL);

std::vector<seal::Ciphertext> HE_encrypt(
        std::vector<std::vector<uint64_t>> &pt,
        const ConvMetadata &data,
        seal::Encryptor &encryptor,
        seal::BatchEncoder &batch_encoder);

std::vector<std::vector<std::vector<seal::Plaintext>>> HE_preprocess_filters_OP(
        Filters &filters,
        const ConvMetadata &data,
        seal::BatchEncoder &batch_encoder);

std::vector<seal::Ciphertext> HE_conv_OP(
        std::vector<std::vector<std::vector<seal::Plaintext>>> &masks,
        std::vector<std::vector<seal::Ciphertext>> &rotations,
        const ConvMetadata &data,
        seal::Evaluator &evaluator,
        seal::Ciphertext &zero);

std::vector<seal::Ciphertext> HE_output_rotations(
        std::vector<seal::Ciphertext> &convs,
        const ConvMetadata &data,
        seal::Evaluator &evaluator,
        seal::GaloisKeys &gal_keys,
        seal::Ciphertext &zero,
        std::vector<seal::Ciphertext> &enc_noise);

uint64_t** HE_decrypt(
        std::vector<seal::Ciphertext> &enc_result,
        const ConvMetadata &data,
        seal::Decryptor &decryptor,
        seal::BatchEncoder &batch_encoder);

class ConvField { 
public:
    int party;
    sci::NetIO* io;
    std::shared_ptr<seal::SEALContext> context[2];
    seal::Encryptor* encryptor[2];
    seal::Decryptor* decryptor[2];
    seal::Evaluator* evaluator[2];
    seal::BatchEncoder* encoder[2];
    seal::GaloisKeys* gal_keys[2];
    seal::Ciphertext* zero[2];
    int slot_count;
    ConvMetadata data;

    ConvField(int party, sci::NetIO* io);

    ~ConvField();

    void configure();

    Image ideal_functionality(
            Image &image,
            Filters &filters);

    void non_strided_conv(
            int32_t H,
            int32_t W,
            int32_t CI,
            int32_t FH,
            int32_t FW,
            int32_t CO,
            Image* image,
            Filters* filters,
            std::vector<std::vector<std::vector<uint64_t>>>& outArr,
            bool verbose = false);
    
    void convolution(
            int32_t N,
            int32_t H,
            int32_t W,
            int32_t CI,
            int32_t FH,
            int32_t FW, 
            int32_t CO,
            int32_t zPadHLeft,
            int32_t zPadHRight,
            int32_t zPadWLeft,
            int32_t zPadWRight,
            int32_t strideH,
            int32_t strideW,
            std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& inputArr,
            std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& filterArr,
            std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& outArr,
            bool verify_output = false,
            bool verbose = false);

    void verify(
            int H,
            int W,
            int CI,
            int CO,
            Image &image,
            Filters* filters,
            std::vector<std::vector<std::vector<std::vector<uint64_t>>>>& outArr);
};


struct OptimizationResult {
    std::vector<int> multiply_indices;
    
    struct AddOp {
        int target_idx;
        int source_idx_1;
        int source_idx_2;
        bool is_const_1;
        bool is_const_2;
    };
    std::vector<AddOp> add_operations;
    
    std::vector<std::pair<int, int>> remaining_nonzero_relations; 
};

class MultiplierOptimizer {
private:
    int out_chans;

public:
    MultiplierOptimizer(int oc) 
        : out_chans(oc) {}

    OptimizationResult optimizeSlice(const std::vector<uint64_t>& slice_data) {
        OptimizationResult result;
        
        std::map<uint64_t, int> unique_nonzero_vals;
        std::map<uint64_t, std::vector<int>> all_nonzero_vals;

        for (int i = 0; i < out_chans; ++i) {
            if (slice_data[i] != 0) {
                all_nonzero_vals[slice_data[i]].push_back(i);
                if (unique_nonzero_vals.find(slice_data[i]) == unique_nonzero_vals.end()) {
                    unique_nonzero_vals[slice_data[i]] = i;
                }
            }
        }

        if (unique_nonzero_vals.empty()) {
            return result;
        }

        std::vector<std::pair<uint64_t, int>> sorted_unique_vals(unique_nonzero_vals.begin(), unique_nonzero_vals.end());
        std::sort(sorted_unique_vals.begin(), sorted_unique_vals.end());

        std::set<uint64_t> ready_multipliers;
        ready_multipliers.insert(1); 
        
        for (auto& [val, orig_idx] : sorted_unique_vals) {

            if (val == 1) {
                result.multiply_indices.push_back(orig_idx);
                continue; 
            }

            bool found_add = false;
            uint64_t k1 = 0, k2 = 0;
            
            for (auto it_a = ready_multipliers.begin(); it_a != ready_multipliers.end(); ++it_a) {
                uint64_t a = *it_a;
                if (a >= val) break;
                
                uint64_t b = val - a;
                if (ready_multipliers.find(b) != ready_multipliers.end()) {
                    k1 = a;
                    k2 = b;
                    found_add = true;
                    break;
                }
            }
            
            if (found_add) {
                OptimizationResult::AddOp op;
                op.target_idx = orig_idx;
                
                op.source_idx_1 = (k1 == 1) ? -1 : unique_nonzero_vals[k1];
                op.source_idx_2 = (k2 == 1) ? -1 : unique_nonzero_vals[k2];
                op.is_const_1 = (k1 == 1);
                op.is_const_2 = (k2 == 1);
                
                result.add_operations.push_back(op);
                
                ready_multipliers.insert(val);
            } else {
                result.multiply_indices.push_back(orig_idx);
                ready_multipliers.insert(val);
            }
        }
        
        for (auto& [val, indices] : all_nonzero_vals) {
            if (indices.size() > 1) {
                int representative_idx = unique_nonzero_vals[val];
                for (size_t i = 1; i < indices.size(); ++i) {
                    int ref_idx = representative_idx;
                    if (val == 1) {
                        ref_idx = -1;
                    }
                    result.remaining_nonzero_relations.push_back({indices[i], ref_idx});
                }
            }
        }
        
        return result;
    }

};


#endif
