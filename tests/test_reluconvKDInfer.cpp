/*
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#define SCI_HE
#define BITLEN_37
#include <iostream>
#include <fstream>
#include <thread>
#include "NonLinear/relu-field.h"
#include "LinearHE/conv-field.h"
#include "globals.h"

//selectively uncomment the following statement if you want to run the code beyond localhost, and then input the specific IP address of the server at the end of "***Argument Parsing***" part
//#define LAN_EXEC
//#define WAN_EXEC

//comment the following statement in ../src/LinearHE/defines-HE.h
//if you want to bypass the verification process: 
//#define DEBUG_EXEC

using namespace sci;
using namespace std;
using namespace seal;

/************* Data Configuration  **********/
/********************************************/
//default Conv structure
int image_h = 28;
int inp_chans = 128;
int filter_h = 1;
int out_chans = 512;
int pad_l = (filter_h-1)/2;
int pad_r = (filter_h-1)/2;
int stride = 1;
//this choice is a trick as analyzed in CTF2
int filter_precision = 12;
//default ReLU and networking configuration
int num_relu = image_h*image_h*inp_chans, port = 32000;
int l = 41, b = 4;
string address;
bool localhost = true;
//set this variable to true to catch up with the computing process
bool verbose_info = true;

//this function performs MSB computing for each thread
void field_relu_thread(int tid, uint64_t* z, uint64_t* x, int lnum_relu) {
    ReLUFieldProtocol<NetIO, uint64_t>* relu_oracle;
    relu_oracle = new ReLUFieldProtocol<NetIO, uint64_t>(party, FIELD, ioArr[tid], l, b, prime_mod, otpackArr[tid]);
    relu_oracle->relu_pregen(z, x, lnum_relu);
    delete relu_oracle;
    return;
}

int main(int argc, char** argv) {

    /************* Argument Parsing  ************/
    /********************************************/
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = server = 1; BOB = client = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("l", l, "Bitlength of inputs");
    //Conv
    amap.arg("h", image_h, "Image Height/Width");
    amap.arg("f", filter_h, "Filter Height/Width");
    amap.arg("i", inp_chans, "Input Channels");
    amap.arg("o", out_chans, "Ouput Channels");
    amap.arg("s", stride, "stride");
    amap.arg("pl", pad_l, "Left Padding");
    amap.arg("pr", pad_r, "Right Padding");
    amap.arg("fp", filter_precision, "Filter Precision");   
    //ReLU
    amap.arg("N", num_relu, "Number of DReLUs");
    //this is the block length namely m that divides the whole bits
    //in millionaries' protocol
    amap.arg("b", b, "Radix base");
    amap.arg("lo", localhost, "Localhost Run?");
    amap.parse(argc, argv);

    if(not localhost) {
#if defined(LAN_EXEC)
        address = "input.your.LAN.address";
#elif defined(WAN_EXEC)
        address = "input.your.WAN.address";
#endif
    } else {
        address = "127.0.0.1";
    }

    //update the numbers based on arg inputs
    num_relu = image_h*image_h*inp_chans;
    pad_l = (filter_h-1)/2;
    pad_r = (filter_h-1)/2;

    cout << "==================================================================" << endl;
    cout << "Role Coding: 1 is server; 2 is client" << endl;
    cout << "==================================================================" << endl;
    cout << "Role: " << party << " - Bitlength: " << bitlength
        << " - Image: " << image_h << "x" << image_h << "x" << inp_chans
        << " - Filter: " << filter_h << "x" << filter_h << "x" << out_chans
        << "\n- Stride: " << stride << "x" << stride
        << " - Padding: " << pad_l << "x" << pad_r
        << " - # Threads: " << numThreads << "\n- Radix Base: " << b
        << " - # DReLUs/MSBs: " << num_relu << endl;
    cout << "==================================================================" << endl;         


    /************ Generate the Data **************/
    /********************************************/
    PRG128 prg;    
    uint64_t *x = new uint64_t[num_relu];//input share
    uint64_t *z = new uint64_t[num_relu];//boolean share
    prg.random_mod_p<uint64_t>(x, num_relu, prime_mod);
    uint64_t *r0;
    Image imageR0;
    //the server generates the share of MSB
    if(party == SERVER){
        bool *g1 = new bool[num_relu];
        prg.random_bool(g1, num_relu);
        for(int j = 0; j< num_relu; j++){
            z[j] = g1[j];
        }
        delete[] g1;
    }
    //define the data sizes
    int newH = 1 + (image_h+pad_l+pad_r-filter_h)/stride;
    int N = 1;
    int W = image_h;
    int FW = filter_h;
    int zPadWLeft = pad_l;
    int zPadWRight = pad_r;
    int strideW = stride;
    int newW = newH;
    int CI = inp_chans;
    int CO = out_chans; 
    
    vector<vector<vector<vector<uint64_t>>>> filterArr(filter_h);
    vector<vector<uint64_t>> outArr(CO);
    vector<vector<uint64_t>> Kr0result(CO);
    Filters myFilters(CO); 
    Filters myFilters_pt(CO); 
    for(int i = 0; i < CO; i++){
        outArr[i].resize(newH * newW);
        for(int j = 0; j < (newH * newW); j++) {
            outArr[i][j] = 0;
        }
    }
    //the server generates its kernel
    if(party == SERVER) {
        for(int i = 0; i < filter_h; i++){
            filterArr[i].resize(FW);
            for(int j = 0; j < FW; j++) {
                filterArr[i][j].resize(CI);
                for(int k = 0; k < CI; k++) {
                    filterArr[i][j][k].resize(CO);
                    prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
                    for(int h = 0; h < CO; h++) {
                        filterArr[i][j][k][h]
                            = ((int64_t) filterArr[i][j][k][h]) >> (64 - filter_precision);
                    }
                }
            }
        }
    }
    //set up the HE context
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    int slot_count =  min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, next_pow2(newH*newH)));
    int num_ct_g1h3 = ceil(1.0 * num_relu / slot_count);
    vector<vector<uint64_t>> shr12off(CO, vector<uint64_t>(slot_count, 0ULL));

    vector<Ciphertext> enc_g1h3;
    shared_ptr<SEALContext> context_;
    Encryptor* encryptor_;
    Decryptor* decryptor_;
    Evaluator* evaluator_;
    BatchEncoder* encoder_;
    GaloisKeys* gal_keys_;
    Ciphertext* zero_;
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);      

    ConvMetadata generalData;
    generalData.inp_chans = CI;
    generalData.image_h = image_h;
    generalData.image_w = W;
    generalData.filter_h = filter_h;
    generalData.filter_w = FW;
    generalData.stride_h = stride;
    generalData.stride_w = strideW;
    generalData.pad_t = pad_l;
    generalData.pad_b = pad_r;
    generalData.pad_l = zPadWLeft;
    generalData.pad_r = zPadWRight;
    generalData.output_h = newH;
    generalData.output_w = newW;    
    generalData.chans_per_cipher = slot_count / (newH * newW);

    int col_widthR0 = generalData.output_h * generalData.output_w;
    int chanPerCipher = generalData.chans_per_cipher;

    /************ Offline Computation ***********/
    /********************************************/
    
    double offComm_total = 0, onComm_total = 0;
    double offTime_total = 0, onTime_total = 0;   

    double offComm_recv = 0, onComm_recv = 0; 
    
    uint64_t offcomm_g1h3 = 0;

    //the server encrypts the g1 and h3
    uint64_t offcomm_start = io->counter;
    auto start_offline = clock_start();    
    if(party == SERVER) {
        vector<Ciphertext> g1h3_ct(2 * num_ct_g1h3);
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_g1h3; i++){
            vector<uint64_t> v1(slot_count, 0ULL);//it's g1
            vector<uint64_t> v2(slot_count, 0ULL);//it's h3
            Plaintext tmp1, tmp2;
            int idx_offset = i * slot_count;
            if(i == (num_ct_g1h3 - 1)){
                for(int j = 0; j < (num_relu - idx_offset); j++){
                    v1[j] = z[idx_offset + j];
                    if ((int64_t)z[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)x[idx_offset + j], prime_mod);
                    }else{
                        v2[j] = neg_mod(-(int64_t)x[idx_offset + j], prime_mod);
                    }
                }
            }else{
                for(int j = 0; j < slot_count; j++){
                    v1[j] = z[idx_offset + j];
                    if ((int64_t)z[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)x[idx_offset + j], prime_mod);
                    }else{
                        v2[j] = neg_mod(-(int64_t)x[idx_offset + j], prime_mod);
                    }
                }
            }
            encoder_->encode(v1, tmp1);
            encoder_->encode(v2, tmp2);
            encryptor_->encrypt(tmp1, g1h3_ct[i]);//it's g1
            evaluator_->mod_switch_to_next_inplace(g1h3_ct[i]);
            encryptor_->encrypt(tmp2, g1h3_ct[num_ct_g1h3 + i]);//it's h3
            evaluator_->mod_switch_to_next_inplace(g1h3_ct[num_ct_g1h3 + i]);
        }    
        //send the cipher to client
        uint64_t g1h3_comm_before = io->counter;
        send_encrypted_vector(io, g1h3_ct);
        uint64_t g1h3_comm_after = io->counter;
        offcomm_g1h3 = g1h3_comm_after - g1h3_comm_before;


        if(verbose_info){
            cout << "[Server] encrypted g1 and h3 sent" << endl;
        }        
        
        Channel k_i2c(CO, (filter_h*FW*CI));
#pragma omp parallel for num_threads(numThreads) schedule(static)         
        for (int out_c = 0; out_c < CO; out_c++) {
            Image tmp_img(CI);
            Image tmp_img1(CI);
            for (int inp_c = 0; inp_c < CI; inp_c++) {
                Channel tmp_chan(filter_h, FW);
                Channel tmp_chan1(filter_h, FW);
                for (int row = 0; row < filter_h; row++) {
                    for (int col = 0; col < FW; col++) {
                        tmp_chan(row, col) = neg_mod(filterArr[row][col][inp_c][out_c], (int64_t)prime_mod);
                        k_i2c(out_c, (inp_c*filter_h*FW+row*FW+col)) = neg_mod((int64_t)filterArr[row][col][inp_c][out_c], prime_mod);
                        tmp_chan1(row, col) = (int64_t)filterArr[row][col][inp_c][out_c];
                    }
                }
                tmp_img[inp_c] = tmp_chan;
                tmp_img1[inp_c] = tmp_chan1;
            }
            myFilters[out_c] = tmp_img;
            myFilters_pt[out_c] = tmp_img1;
        }
        
        //recieve r0hat
        const int col_heightR0 = generalData.filter_h * generalData.filter_w * generalData.inp_chans;
        // const int col_widthR0 = generalData.output_h * generalData.output_w;
        const int f_size = generalData.filter_h * generalData.filter_w;
        // int chanPerCipher = generalData.chans_per_cipher;
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / 1); //one channel for one input
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        }        
        
        //perform rot-free computation with plaintext kernel
        vector<Ciphertext> enc_Kr0(CO);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < col_heightR0; i++){       
            //get one col in 2-dim kernel
            vector<uint64_t> k_row(CO, 0ULL);
            vector<Ciphertext> tp_Kr0(CO);
            int nonzero_num = 0;
            Ciphertext const_cipher = enc_r0hat[i];
            for(int j = 0; j < CO; j++){
                k_row[j] = k_i2c(j, i);
                if(k_i2c(j, i) != 0)nonzero_num++;
                if(i == 0){
                    enc_Kr0[j] = *zero_;
                    evaluator_->mod_switch_to_next_inplace(enc_Kr0[j]);
                }
            }
                        
            //get the mult and add information
            MultiplierOptimizer optimizer(CO);
            auto res = optimizer.optimizeSlice(k_row);

            int deal_num = 0;
            //first deal with the mult
            if (!(res.multiply_indices.empty())) {
                for (size_t j = 0; j < res.multiply_indices.size(); ++j) {
                    auto tp = res.multiply_indices[j];
                    if(k_row[tp] != 1){
                        vector<uint64_t> k_tmp(slot_count, k_row[tp]);
                        Plaintext pt_tmp;
                        encoder_->encode(k_tmp, pt_tmp);
                        Ciphertext tmp_ct;
                        evaluator_->multiply_plain(enc_r0hat[i], pt_tmp, tmp_ct);
                        tp_Kr0[tp] = tmp_ct;
                        evaluator_->add_inplace(enc_Kr0[tp], tmp_ct);
                        deal_num++;
                    }else if (k_row[tp] == 1){
                        tp_Kr0[tp] = const_cipher;
                        evaluator_->add_inplace(enc_Kr0[tp], const_cipher);
                        deal_num++;
                    }
                    
                }
            }

            //then deal with the add
            if (!(res.add_operations.empty())) {
                for (const auto& op : res.add_operations) {
                    int idx_tp = op.target_idx;
                    Ciphertext ct_1;
                    if (op.is_const_1){
                        ct_1 = const_cipher;
                    }else{
                        ct_1 = tp_Kr0[op.source_idx_1];
                    }
                    
                    if (op.is_const_2){
                        evaluator_->add_inplace(ct_1, const_cipher);
                    }else{
                        evaluator_->add_inplace(ct_1, tp_Kr0[op.source_idx_2]);
                    }
                    
                    tp_Kr0[idx_tp] = ct_1;
                    evaluator_->add_inplace(enc_Kr0[idx_tp], ct_1);
                    deal_num++;
                }
            }

            //the rest non-zero items
            if (!(res.remaining_nonzero_relations.empty())) {
                for (const auto& rel : res.remaining_nonzero_relations) {
                    if (rel.second == -1){
                        evaluator_->add_inplace(enc_Kr0[rel.first], const_cipher);
                    }else{
                        evaluator_->add_inplace(enc_Kr0[rel.first], tp_Kr0[rel.second]);
                    }
                    deal_num++;
                }
            }

            //make sure all non-zero items are dealt with
            assert(deal_num == nonzero_num && "some items are missing in adder tree!");

        }

        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < CO; i++){
            //add the noise
            prg.random_mod_p<uint64_t>(shr12off[i].data(), slot_count, prime_mod);
            Plaintext tmp_res;
            encoder_->encode(shr12off[i], tmp_res);
            evaluator_->add_plain_inplace(enc_Kr0[i], tmp_res);
        }

        
        //perform the noise flooding
        parms_id_type parms_id = enc_Kr0[0].parms_id();
        shared_ptr<const SEALContext::ContextData> context_data
        = context_->get_context_data(parms_id);        
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int ct_idx = 0; ct_idx < CO; ct_idx++) {
            flood_ciphertext(enc_Kr0[ct_idx], context_data, SMUDGING_BITLEN);
            evaluator_->mod_switch_to_next_inplace(enc_Kr0[ct_idx]);
        }
        
        //send masked kR0hat
        send_encrypted_vector(io, enc_Kr0);

#if defined(DEBUG_EXEC)
        GET_NOISE_BUDGET(decryptor_, enc_Kr0[0], "Server", "after mod-switch");
#endif            
        if(verbose_info){
            cout << "[Server] encrypted share sent" << endl;
        }

        //reset the shr12off
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < (chanPerCipher * col_widthR0); j++){
                shr12off[i][j] = neg_mod((int64_t)(prime_mod - shr12off[i][j]), prime_mod);
            }
        }
        
        //gets shr1
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_widthR0; j++){
                outArr[i][j] = shr12off[i][j];
            }
        }
        
        if(verbose_info){
            cout << "[Server] all share generated" << endl;
        }

        
    }else{//the client encrypts r0 and decrypts the result
        //generate r0
        r0 = new uint64_t[num_relu];
        prg.random_mod_p<uint64_t>(r0, num_relu, prime_mod);        
        //transform r0 into r0hat and perform encryption
        imageR0.resize(CI);
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)r0[idx], prime_mod);
                }
            }
            imageR0[chan] = tmp_chan;
        }        
        //transform r0
        auto p_imageR0 = pad_image(generalData, imageR0);
        const int col_heightR0 = generalData.filter_h * generalData.filter_w * generalData.inp_chans;
        // const int col_widthR0 = generalData.output_h * generalData.output_w;
        Channel image_colR0(col_heightR0, col_widthR0);
        i2c(p_imageR0, image_colR0, generalData.filter_h, generalData.filter_w, generalData.stride_h, generalData.stride_w, generalData.output_h, generalData.output_w);
              
        //encrypt r0hat
        // int chanPerCipher = generalData.chans_per_cipher;
        int cipherNum = ceil(1.0 * col_heightR0 / 1); //one channel for one input
        vector<Ciphertext> r0hat_ct(cipherNum);        
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < cipherNum; i++){
            vector<uint64_t> tmp_vec(slot_count, 0ULL);
            Plaintext tmp_pt;
            for(int k = 0; k < col_widthR0; k++){
                tmp_vec[k] = image_colR0(i, k);
            }
            
            //encrypt the plaintext vector
            encoder_->encode(tmp_vec, tmp_pt);
            encryptor_->encrypt(tmp_pt, r0hat_ct[i]);
            evaluator_->mod_switch_to_next_inplace(r0hat_ct[i]);
        }
        
        //recieve g1 and h3
        enc_g1h3.resize(2 * num_ct_g1h3);
        recv_encrypted_vector(io, enc_g1h3);

        if(verbose_info){
            cout << "[Client] encrypted g1 and h3 received" << endl;
        }

        
        //send the encrypted r0hat
        send_encrypted_vector(io, r0hat_ct);
        
        if(verbose_info){
            cout << "[Client] encrypted r0 hat sent" << endl;
        }
        
        //recieve the masked kR0
        vector<Ciphertext> ct_Kr0(CO);
        recv_encrypted_vector(io, ct_Kr0);
        
        if(verbose_info){
            cout << "[Client] masked Kr0 received" << endl;
        }
        
        //decrypt the masked kR0
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for (int ct_idx = 0; ct_idx < CO; ct_idx++) {
            Plaintext tmp;
            Kr0result[ct_idx].resize(slot_count);
            decryptor_->decrypt(ct_Kr0[ct_idx], tmp);
            encoder_->decode(tmp, Kr0result[ct_idx]);
        }
        if(verbose_info){cout << "[Client] share decrypted" << endl;}        

    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "Comm. Sent at offline per input (MiB): " << (offcomm_end - offcomm_start - offcomm_g1h3)/(1.0*(1ULL << 20)*chanPerCipher) + (offcomm_g1h3)/(1.0*(1ULL << 20)) << endl;
    cout <<"Offline Time per input (l=" << l << "; b=" << b << ") " << t_off * 1.0 / (1000*chanPerCipher) <<" ms"<< endl;

    offComm_total += ((offcomm_end - offcomm_start - offcomm_g1h3)/(1.0*(1ULL << 20)*chanPerCipher) + (offcomm_g1h3)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / (1000*chanPerCipher));    

    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)*chanPerCipher));
        //cout << "Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }


    //verify the result
#if defined(DEBUG_EXEC)        
    if(party == SERVER) {
        Image image_0(CI);
        for(int i = 0; i < CI; i++) {
            image_0[i].resize(image_h, W);
            io->recv_data(image_0[i].data(), image_h * W * sizeof(uint64_t));
        }   

        Image result = ideal_function(image_0, myFilters_pt, generalData);
        
        //the share from client
        vector<vector<uint64_t>> shrArr_0;
        shrArr_0.resize(CO);
        for(int i = 0; i < CO; i++) {
            shrArr_0[i].resize(slot_count);
            io->recv_data(shrArr_0[i].data(), sizeof(uint64_t) * slot_count);
        }
        

        for(int i = 0; i < CO; i++) {
            for(int j = 0; j < col_widthR0; j++) {
                shrArr_0[i][j] = (shrArr_0[i][j] + shr12off[i][j]) % prime_mod;
                shrArr_0[i][j] = neg_mod((int64_t)shrArr_0[i][j], prime_mod);
            }
        }
        
        bool pass = true;
        for (int i = 0; i < CO; i++) {
            for (int j = 0; j < col_widthR0; j++) {
                int tmp_row = j / generalData.output_w;
                int tmp_col = j % generalData.output_w;
                if (shrArr_0[i][j] != neg_mod((int64_t)result[i](tmp_row, tmp_col), prime_mod)){
                    pass = false;
                }
            }
        }
        if (pass) {
            cout << GREEN << "[Server] Successful Offline" << RESET << endl;
        }
        else {
            cout << RED << "[Server] Failed Offline" << RESET << endl;
            cout << RED << "WARNING: The implementation assumes that the computation" << endl;
            cout << "performed by the server (on it's model and client-encrypted r0)" << endl;
            cout << "fits in a 64-bit integer. The failed operation could be a result" << endl;
            cout << "of overflowing the bound." << RESET << endl;
        }
    }else{

        //verify the result
        for(int i = 0; i < CI; i++) {
            io->send_data(imageR0[i].data(), image_h * W * sizeof(uint64_t));
        }
        for(int i = 0; i < CO; i++) {
            io->send_data(Kr0result[i].data(), sizeof(uint64_t) * slot_count);
        }        

    }

#endif 

    io->flush();

    delete io;

    cout <<"==================================================================" << endl;


    /******** Online Computation: OT Part********/
    /********** Setup IO and Base OTs ***********/
    /********************************************/
    for(int i = 0; i < numThreads; i++) {
        ioArr[i] = new NetIO(party==1 ? nullptr:address.c_str(), port+i);
        if (i == 0) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l);
        }else {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l, false);
            otpackArr[i]->copy(otpackArr[0]);
        }
    }
    if(verbose_info) std::cout << "All Base OTs Done" << std::endl;

    /******** Online Computation: OT Part********/	
    /************** Fork Threads ****************/
    /********************************************/
    uint64_t comm_sent = 0;
	uint64_t multiThreadedIOStart[numThreads];
	for(int i=0;i<numThreads;i++){
		multiThreadedIOStart[i] = ioArr[i]->counter;
	}
    auto start = clock_start();
    std::thread relu_threads[numThreads];
    int chunk_size = num_relu/numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int offset = i*chunk_size;
        int lnum_relu;
        if (i == (numThreads - 1)) {
            lnum_relu = num_relu - offset;
        } else {
            lnum_relu = chunk_size;
        }
        relu_threads[i] = std::thread(field_relu_thread, i, z+offset, x+offset, lnum_relu);
    }
    for (int i = 0; i < numThreads; ++i) {
      relu_threads[i].join();
    }
    long long t = time_from(start);
	for(int i=0;i<numThreads;i++){
		auto curComm = (ioArr[i]->counter) - multiThreadedIOStart[i];
		comm_sent += curComm;
	}
    cout <<"Comm. Sent for MSB (MiB): " << double(comm_sent)/(1.0*(1ULL<<20)) << std::endl;
    cout <<"Online Time for MSB (l=" << l << "; b=" << b << ") " << t * 1.0 / 1000 <<" ms"<< endl;

    onComm_total += (double(comm_sent)/(1.0*(1ULL<<20)));
    onTime_total += (t * 1.0 / 1000);

    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = comm_sent;
        ioArr[0]->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        ioArr[0]->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Comm. Sent & Recv-ed for MSB (MiB): " << (comm_sent + myRecev)/(1.0*(1ULL << 20)) << endl;
    }



    /******** Online Computation: OT Part********/
    /************** Verification ****************/
    /********************************************/
#if defined(DEBUG_EXEC)
    switch (party) {
        case sci::ALICE: {
            ioArr[0]->send_data(x, sizeof(uint64_t) * num_relu);
            ioArr[0]->send_data(z, sizeof(uint64_t) * num_relu);
            break;
        }
        case sci::BOB: {
            uint64_t *xi = new uint64_t[num_relu];
            uint64_t *zi = new uint64_t[num_relu];
            ioArr[0]->recv_data(xi, sizeof(uint64_t) * num_relu);
            ioArr[0]->recv_data(zi, sizeof(uint64_t) * num_relu);
            
            for(int i=0; i<num_relu; i++){
                xi[i] = (xi[i] + x[i]) % prime_mod;
                zi[i] = (zi[i] + z[i]) % 2;//this recovers the MSB from two boolean shares
                assert((zi[i] == (xi[i] > prime_mod/2))
                        && "MSB protocol's answer is incorrect!");
            }
            cout << GREEN << "[Client] Successful MSB Computing" << RESET << endl;
            delete[] xi;
            delete[] zi;
            break;
        }
    }
#endif

    
    /******** Online Computation: OT Part********/
    /******************* Cleanup ****************/
    /********************************************/
    for (int i = 0; i < numThreads; i++) {
        delete ioArr[i];
        delete otpackArr[i];
    }


    /******** Online Computation: HE Part********/
    /********************************************/    
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);    

    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();  
    
    if(party == CLIENT){
        vector<Ciphertext> h5_ct(num_ct_g1h3);
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_g1h3; i++){
            h5_ct[i] = *zero_;
            evaluator_->mod_switch_to_next_inplace(h5_ct[i]);
            vector<uint64_t> v1(slot_count, 0ULL);//it's g0
            vector<uint64_t> v2(slot_count, 0ULL);//it's h1
            vector<uint64_t> v3(slot_count, 0ULL);//it's h2
            Plaintext tmp1, tmp2, tmp3;
            int idx_offset = i * slot_count;
            if(i == (num_ct_g1h3 - 1)){
                for(int j = 0; j < (num_relu - idx_offset); j++){
                    v1[j] = (z[idx_offset + j] ^ 1);//the drelu share
                    if ((int64_t)z[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)x[idx_offset + j] - (int64_t)r0[idx_offset + j], prime_mod);//it's h1
                        v3[j] = neg_mod(-(int64_t)x[idx_offset + j], prime_mod);//it's h2
                    }else{
                        v2[j] = neg_mod(-(int64_t)r0[idx_offset + j], prime_mod);
                        v3[j] = neg_mod((int64_t)x[idx_offset + j], prime_mod);
                    }
                }
            }else{
                for(int j = 0; j < slot_count; j++){
                    v1[j] = (z[idx_offset + j] ^ 1);
                    if ((int64_t)z[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)x[idx_offset + j] - (int64_t)r0[idx_offset + j], prime_mod);//it's h1
                        v3[j] = neg_mod(-(int64_t)x[idx_offset + j], prime_mod);//it's h2
                        
                    }else{
                        v2[j] = neg_mod(-(int64_t)r0[idx_offset + j], prime_mod);
                        v3[j] = neg_mod((int64_t)x[idx_offset + j], prime_mod);
                    }
                }
            }
            encoder_->encode(v1, tmp1);
            encoder_->encode(v2, tmp2);
            encoder_->encode(v3, tmp3);
            Ciphertext tmp_ct1, tmp_ct2;
            //multiply h2 with g1
            evaluator_->multiply_plain(enc_g1h3[i], tmp3, tmp_ct1);

            //multiply h3 with g0
            evaluator_->multiply_plain(enc_g1h3[i + num_ct_g1h3], tmp1, tmp_ct2);
            
            //add up the terms
            evaluator_->add_inplace(h5_ct[i], tmp_ct1);
            evaluator_->add_inplace(h5_ct[i], tmp_ct2);
            evaluator_->add_plain_inplace(h5_ct[i], tmp2);
            
        }
        
        //perform the noise flooding
        parms_id_type parms_id = h5_ct[0].parms_id();
        shared_ptr<const SEALContext::ContextData> context_data = context_->get_context_data(parms_id);        
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int ct_idx = 0; ct_idx < num_ct_g1h3; ct_idx++) {
            flood_ciphertext(h5_ct[ct_idx], context_data, SMUDGING_BITLEN);
            evaluator_->mod_switch_to_next_inplace(h5_ct[ct_idx]);
        }
        
        //send h5
        send_encrypted_vector(io, h5_ct);

        if(verbose_info){
            cout << "[Client] encrypted h5 sent" << endl;
        }        
        

#if defined(DEBUG_EXEC)
        GET_NOISE_BUDGET(decryptor_, h5_ct[0], "Client", "after mod-switch");
#endif        
          
        
        //get shr0
        for(int i = 0; i < CO; i++) {
            for(int j = 0; j < col_widthR0; j++) {
                outArr[i][j] = Kr0result[i][j];
                outArr[i][j] = neg_mod((int64_t)outArr[i][j], prime_mod);
            }
        }
    
    }else{//the server decrypts the data and gets share
    
        //receive the h5
        vector<Ciphertext> enc_h5(num_ct_g1h3);
        recv_encrypted_vector(io, enc_h5);
    
        if(verbose_info){
            cout << "[Server] encrypted h5 received" << endl;
        }
        
        //long long t_decs = time_from(start_online);
        
        //decrypt the h5
        vector<vector<uint64_t>> pt_h5(num_ct_g1h3);
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for (int ct_idx = 0; ct_idx < num_ct_g1h3; ct_idx++) {
            Plaintext tmp;
            pt_h5[ct_idx].resize(slot_count);
            decryptor_->decrypt(enc_h5[ct_idx], tmp);
            encoder_->decode(tmp, pt_h5[ct_idx]);
        }
        if(verbose_info){cout << "[Server] share decrypted" << endl;}
        
        
        //long long t_dece = time_from(start_online);
        //if(verbose_info){cout <<"[Server] Dec Time (l=" << l << "; b=" << b << ") " << (t_dece - t_decs) * 1.0 / 1000 <<" ms"<< endl;}
        

        //form the image
        Image imageH5(CI);
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)pt_h5[idx / slot_count][idx % slot_count], prime_mod);
                    if((int64_t)z[idx] == 1){
                        uint64_t tp = neg_mod((int64_t)x[idx], prime_mod);
                        tmp_chan(h, w) = (tmp_chan(h, w) + tp) % prime_mod;
                        tmp_chan(h, w) = neg_mod((int64_t)tmp_chan(h, w), prime_mod);
                    }
                }
            }
            imageH5[chan] = tmp_chan;
        }
        
        //do the convolution and the filter values should
        //be small enough to fit uint64_t
        Image local_kH5 = ideal_function(imageH5, myFilters_pt, generalData);
        
        //form the share
        const int col_w = generalData.output_h * generalData.output_w;
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_w; j++){
                outArr[i][j] = neg_mod((int64_t)outArr[i][j] + (int64_t)local_kH5[i](j / generalData.output_w, j % generalData.output_w), prime_mod);    
            }
        
        }       
    
        if(verbose_info){cout << "[Server] share formed" << endl;}
    
    }
    
    long long t_on = time_from(start_online);
    uint64_t oncomm_end = io->counter;
    cout << "Comm. Sent after MSB (MiB): " << (oncomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"Online Time after MSB (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    
    onComm_total += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);    

    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (oncomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Comm. Sent & Recv-ed after MSB (MiB): " << (oncomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }


        //verify the result
#if defined(DEBUG_EXEC)      
    if(party == CLIENT) {
    //form the input image
        Image imageInp(CI);
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)x[idx], prime_mod);
                }
            }
            imageInp[chan] = tmp_chan;
        }
        
        //send input share
        for(int i = 0; i < CI; i++) {
            io->send_data(imageInp[i].data(), image_h * W * sizeof(uint64_t));
        }
        
        //send MSB share
        io->send_data(z, sizeof(uint64_t) * num_relu); 
        
        //send final share
        for(int i = 0; i < CO; i++) {
            io->send_data(outArr[i].data(), sizeof(uint64_t) * col_widthR0);
        }   
        
    }else{
        //verify the result    
        //receive input share
        Image image_in(CI);
        for(int i = 0; i < CI; i++) {
            image_in[i].resize(image_h, W);
            io->recv_data(image_in[i].data(), image_h * W * sizeof(uint64_t));
        }   

        //receive MSB share
        uint64_t *zi = new uint64_t[num_relu];
        io->recv_data(zi, sizeof(uint64_t) * num_relu);

        //form the input image
        for(int i = 0; i < CI; i++) {
            for(int h = 0; h < image_h; h++) {
                for(int w = 0; w < W; w++) {
                    int idx = i * image_h * W + h * W + w;
                    image_in[i](h,w) = (neg_mod((int64_t)x[idx], prime_mod) + image_in[i](h,w)) % prime_mod;
                    int drelu_tmp = (z[idx] + zi[idx] + 1) % 2;
                    image_in[i](h,w) = image_in[i](h,w) * drelu_tmp;
                }
            }
        }
        
        //get the convolution
        Image resultConv = ideal_function(image_in, myFilters_pt, generalData);

        //receive final share
        vector<vector<uint64_t>> outArr_0;
        outArr_0.resize(CO);
        for(int i = 0; i < CO; i++) {
            outArr_0[i].resize(col_widthR0);
            io->recv_data(outArr_0[i].data(), sizeof(uint64_t) * col_widthR0);
        }

        //get the result from final shares
        for(int i = 0; i < CO; i++) {
            for(int j = 0; j < col_widthR0; j++) {
                outArr_0[i][j] = (outArr_0[i][j] + outArr[i][j]) % prime_mod;
            }
        }

        //compare the result
        bool pass = true;
        for (int i = 0; i < CO; i++) {
            for (int j = 0; j < newH; j++) {
                for (int k = 0; k < newW; k++) {
                    int idx = j * newW + k;
                    if (outArr_0[i][idx] != neg_mod(resultConv[i](j,k), (int64_t) prime_mod)){
                        pass = false;
                    }
                }
            }
        }

        if (pass) {
            cout << GREEN << "[Server] Successful Online" << RESET << endl;
        }
        else {
            cout << RED << "[Server] Failed Online" << RESET << endl;
            cout << RED << "WARNING: The implementation assumes that the computation" << endl;
            cout << "performed by the server (on it's model and h5)" << endl;
            cout << "fits in a 64-bit integer. The failed operation could be a result" << endl;
            cout << "of overflowing the bound." << RESET << endl;
        }

        delete[] zi;

    }
#endif


    cout <<"==================================================================" << endl;
    cout << "Each Input(offline): Comm. Sent Offline (MiB): " << offComm_total << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed Offline (MiB): " << (offComm_total + offComm_recv) << endl;
    }    
    cout << "          : Offline Time (l=" << l << "; b=" << b << ") " << offTime_total <<" ms"<< endl; 
    cout <<"------------------------------------------------------------------" << endl;
    cout << "Each Input(online): Comm. Sent Online (MiB): " << onComm_total << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed Online (MiB): " << (onComm_total + onComm_recv) << endl;
    }
    cout << "          : Online Time (l=" << l << "; b=" << b << ") " << onTime_total <<" ms"<< endl;        
    cout <<"------------------------------------------------------------------" << endl;
    cout << "Each Input(total): Comm. Sent (MiB): " << (offComm_total + onComm_total) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (offComm_total + offComm_recv + onComm_total + onComm_recv) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (offTime_total + onTime_total) <<" ms"<< endl;    
    cout <<"==================================================================" << endl;
    
    io->flush();    
    
    delete io;
    
    //clean the data
    delete[] x;
    delete[] z;
    if (party == CLIENT){
        delete[] r0; 
    }
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
       
	return 0;
}
