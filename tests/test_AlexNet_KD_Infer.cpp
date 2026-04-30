#define SCI_HE
#define BITLEN_37
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <thread>
#include "NonLinear/relu-field.h"
#include "LinearHE/conv-field.h"
#include "functionalities.h"
#include "NonLinear/maxpool.h"
#include "NonLinear/argmax.h"

//selectively uncomment the following statement if you want to run the code beyond localhost, and then input the specific IP address of the server at the end of "***Address Config***" part
//#define LAN_EXEC
//#define WAN_EXEC

//comment the following statement in ../src/LinearHE/defines-HE.h
//if you want to bypass the verification process: 
//#define DEBUG_EXEC

using namespace sci;
using namespace std;
using namespace seal;

/************* Data Configuration ***********/
//default ReLU and networking configuration
int l = bitlength, b = 4;
string address;
bool localhost = true;
int port = 32000;
//set this variable to true to catch up with the computing process
bool verbose_info = true;
PRG128 prg;
/********************************************/

//this function performs MSB computing for each thread
void field_relu_thread(uint64_t* z, uint64_t* x, int lnum_relu);


void sole_relu(uint64_t* relu_z, uint64_t* inp_shrs, int number_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_relu, double &onTime_relu, double &onComm_recv_relu);

void sole_div(int div_num, int divisor, int div_const, uint64_t* avrg_pool_l, uint64_t* outp_final_l, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_trunc, double &onTime_trunc, double &onComm_recv_trunc);

void first_conv_out(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, string input_addr, string wts_addr, string bs_addr, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_soleconv, double &Time_soleconv, double &Comm_soleconv_recv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void mxp(uint64_t *x, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_mxp, double &onTime_mxp, double &onComm_recv_mxp);

void trunc_func(int div_num, int consSF, uint64_t *outp_final, uint64_t *x, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_trunc, double &onTime_trunc, double &onComm_recv_trunc);

void reconv_normal(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, vector<vector<uint64_t>> &outArr_two, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void final_fc_max(uint64_t *outp_final, int x_scales, int inp_dim, int out_dim, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_solefc, double &Time_solefc, double &Comm_solefc_recv, double &Comm_argmax, double &Time_argmax, double &Comm_argmax_recv);

void sole_fc(uint64_t *outp_final, uint64_t *outFC, int col_start, int col_end, int x_scales, int inp_dim, int out_dim, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_solefc, double &Time_solefc, double &Comm_solefc_recv);




int main(int argc, char** argv) {

    /************* Argument Parsing  ************/
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = server = 1; BOB = client = 2");
    amap.arg("p", port, "Port Number");
    //the m in CTF2 millionaries' protocol
    amap.arg("b", b, "Radix base");
    amap.arg("lo", localhost, "Localhost Run?");
    amap.parse(argc, argv);
    /********************************************/

    /************* Address Config  ************/
    if(not localhost) {
        #if defined(LAN_EXEC)
            address = "input.your.LAN.address";
        #elif defined(WAN_EXEC)
            address = "input.your.WAN.address";
        #endif
    } else {
        address = "127.0.0.1";
    }
    /********************************************/

    /************* Print Role Info  ************/
    string pary_str = "server";
    if(party == CLIENT){
        pary_str = "client";
    }
    cout << "==========" << endl;
    cout << "Role: " << pary_str << " - # Threads: " << numThreads << endl;
    cout << "==========" << endl; 
    /********************************************/

    /************ Some global variables ***************/
    double offComm_total_g = 0, onComm_total_g = 0;
    double offTime_total_g = 0, onTime_total_g = 0;   
    double offComm_recv_g = 0, onComm_recv_g = 0;
    double Comm_relu_g = 0, Time_relu_g = 0, Comm_recv_relu_g = 0;
    double Comm_trunc_g = 0, Time_trunc_g = 0, Comm_recv_trunc_g = 0;
    double Comm_soleconv_g = 0, Time_soleconv_g = 0, Comm_recv_soleconv_g = 0;
    double Comm_mxp_g = 0, Time_mxp_g = 0, Comm_recv_mxp_g = 0;
    double Comm_solefc_g = 0, Time_solefc_g = 0, Comm_recv_solefc_g = 0;
    double Comm_argmax_g = 0, Time_argmax_g = 0, Comm_argmax_recv_g = 0;
    ConvMetadata generalData_g;
  
    string model_wts_root = "../../src/ModelAndInput/processed_weights_and_biases_alexnet/";
    string model_inp_root = "../../src/ModelAndInput/";

    /************ Prepare conv 1 ***************/
    vector<vector<uint64_t>> outArr_g;
    first_conv_out(224, 3, 11, 64, 2, 2, 4, model_inp_root+"uint64_inp.bin", model_wts_root+"features.0.weight.bin", model_wts_root+"features.0.bias.bin", generalData_g, outArr_g, offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_soleconv_g, Time_soleconv_g, Comm_recv_soleconv_g);


    /***************maxpooling*******************/
    generalData_g.inp_chans = 64;
    generalData_g.image_h = 55;
    generalData_g.image_w = 55;
    generalData_g.filter_h = 3;
    generalData_g.filter_w = 3;
    generalData_g.stride_h = 2;
    generalData_g.stride_w = 2;
    generalData_g.pad_t = 1;
    generalData_g.pad_b = 1;
    generalData_g.pad_l = 1;
    generalData_g.pad_r = 1;
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 64;
    int mxpool_row = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    uint64_t *x_g = new uint64_t[mxpool_row];
    mxp(x_g, generalData_g, outArr_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_mxp_g, Time_mxp_g, Comm_recv_mxp_g);
    outArr_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 64;
    int div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    int consSF_g = 12;
    uint64_t *outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;
   


    /************ Prepare conv 2***************/  
    reconv_normal(28, 64, 5, 192, 2, 2, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"features.3.weight.bin", model_wts_root+"features.3.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;   


    /***************maxpooling*******************/
    generalData_g.inp_chans = 192;
    generalData_g.image_h = 28;
    generalData_g.image_w = 28;
    generalData_g.filter_h = 3;
    generalData_g.filter_w = 3;
    generalData_g.stride_h = 2;
    generalData_g.stride_w = 2;
    generalData_g.pad_t = 3;
    generalData_g.pad_b = 2;
    generalData_g.pad_l = 3;
    generalData_g.pad_r = 2;
    generalData_g.output_h = 16;
    generalData_g.output_w = 16;
    generalData_g.out_chans = 192;
    mxpool_row = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    x_g = new uint64_t[mxpool_row];
    mxp(x_g, generalData_g, outArr_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_mxp_g, Time_mxp_g, Comm_recv_mxp_g);
    outArr_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 16;
    generalData_g.output_w = 16;
    generalData_g.out_chans = 192;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;       
    

    /************ Prepare conv 3***************/
    reconv_normal(16, 192, 3, 384, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"features.6.weight.bin", model_wts_root+"features.6.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;    
    

    /***************truncation*******************/
    generalData_g.output_h = 16;
    generalData_g.output_w = 16;
    generalData_g.out_chans = 384;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_g = vector<vector<uint64_t>>();


    /************ Prepare conv 4***************/
    reconv_normal(16, 384, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"features.8.weight.bin", model_wts_root+"features.8.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /***************truncation*******************/
    generalData_g.output_h = 16;
    generalData_g.output_w = 16;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }

    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;  
    outArr_g = vector<vector<uint64_t>>();


    /************ Prepare conv 5***************/
    reconv_normal(16, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"features.10.weight.bin", model_wts_root+"features.10.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;    
    

    /***************maxpooling*******************/
    generalData_g.inp_chans = 256;
    generalData_g.image_h = 16;
    generalData_g.image_w = 16;
    generalData_g.filter_h = 3;
    generalData_g.filter_w = 3;
    generalData_g.stride_h = 2;
    generalData_g.stride_w = 2;
    generalData_g.pad_t = 1;
    generalData_g.pad_b = 1;
    generalData_g.pad_l = 1;
    generalData_g.pad_r = 1;
    generalData_g.output_h = 8;
    generalData_g.output_w = 8;
    generalData_g.out_chans = 256;
    mxpool_row = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    x_g = new uint64_t[mxpool_row];
    mxp(x_g, generalData_g, outArr_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_mxp_g, Time_mxp_g, Comm_recv_mxp_g);
    outArr_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 8;
    generalData_g.output_w = 8;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;


    /**************get the relu******************/
    generalData_g.output_h = 8;
    generalData_g.output_w = 8;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    uint64_t *relu_g = new uint64_t[div_num_g];
    sole_relu(relu_g, outp_final_g, div_num_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);
    delete[] outp_final_g;
    outp_final_g = nullptr;
    

    /**************tailor the size***************/    
    outp_final_g = new uint64_t[9216];
    for(int i = 0; i < 256; i++){
        for(int j = 0; j < 6; j++){
            for(int k = 0; k < 6; k++){
                outp_final_g[i*36 + j*6 + k] = relu_g[i*64 + (j+1)*8 + (k+1)];
            }
        }
    }

    delete[] relu_g;
    relu_g = nullptr;


    /****************FC-1 output*******************/
    int out_dim = 4096;
    uint64_t *outFC = new uint64_t[out_dim];
    for(int i = 0; i < out_dim; i++){
        outFC[i] = 0;
    }
    int in_dim = 9216;
    int num_pieces = 4;
    int piece_size = in_dim / num_pieces;

    for(int i = 0; i < num_pieces; i++){
        int col_start_g = i*piece_size;
        int col_end_g = (i+1)*piece_size - 1;
        uint64_t *outFC_local = new uint64_t[out_dim];
        sole_fc(outp_final_g, outFC_local, col_start_g, col_end_g, 12, 9216, 4096, model_wts_root+"classifier.1.weight.bin", model_wts_root+"classifier.1.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_solefc_g, Time_solefc_g, Comm_recv_solefc_g);
        for(int j = 0; j < out_dim; j++){
            outFC[j] = (outFC_local[j] + outFC[j]) % prime_mod;
            outFC[j] = neg_mod((int64_t)outFC[j], prime_mod);
        }
        delete[] outFC_local;
        outFC_local = nullptr;
    }
    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    div_num_g = 4096;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, outFC, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] outFC;
    outFC = nullptr; 


    /**************get the relu******************/
    div_num_g = 4096;
    relu_g = new uint64_t[div_num_g];
    sole_relu(relu_g, outp_final_g, div_num_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);
    delete[] outp_final_g;
    outp_final_g = nullptr;


    /****************FC-2 output*******************/
    out_dim = 4096;
    outFC = new uint64_t[out_dim];
    sole_fc(relu_g, outFC, 0, 4095, 12, 4096, 4096, model_wts_root+"classifier.4.weight.bin", model_wts_root+"classifier.4.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_solefc_g, Time_solefc_g, Comm_recv_solefc_g);

    delete[] relu_g;
    relu_g = nullptr;


    /***************truncation*******************/
    div_num_g = 4096;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, outFC, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] outFC;
    outFC = nullptr; 


    /**************get the relu******************/
    div_num_g = 4096;
    relu_g = new uint64_t[div_num_g];
    sole_relu(relu_g, outp_final_g, div_num_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);
    delete[] outp_final_g;
    outp_final_g = nullptr;



    /****************FC output*******************/
    final_fc_max(relu_g, 12, 4096, 1000, model_wts_root+"classifier.6.weight.bin", model_wts_root+"classifier.6.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_solefc_g, Time_solefc_g, Comm_recv_solefc_g, Comm_argmax_g, Time_argmax_g, Comm_argmax_recv_g);

    delete[] relu_g;
    relu_g = nullptr;


    //print overall time and communication
    cout <<"========================" << endl;
    cout << "Each Input(offline): Comm. Sent Offline (MiB): " << offComm_total_g << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed Offline (MiB): " << (offComm_total_g + offComm_recv_g) << endl;
    }    
    cout << "          : Offline Time (l=" << l << "; b=" << b << ") " << offTime_total_g <<" ms"<< endl; 
    cout <<"------------------------" << endl;
    cout << "Each Input(online): Comm. Sent Online (MiB): " << onComm_total_g << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed Online (MiB): " << (onComm_total_g + onComm_recv_g) << endl;
    }
    cout << "          : Online Time (l=" << l << "; b=" << b << ") " << onTime_total_g <<" ms"<< endl;        
    cout <<"------------------------" << endl;
    cout << " Each Input(total): Comm. Sent (MiB): " << (offComm_total_g + onComm_total_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (offComm_total_g + offComm_recv_g + onComm_total_g + onComm_recv_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (offTime_total_g + onTime_total_g) <<" ms"<< endl;   
    cout <<"------------------------" << endl;
    cout << " Each Input(sole relu): Comm. Sent (MiB): " << (Comm_relu_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_relu_g + Comm_recv_relu_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_relu_g) <<" ms"<< endl;    
    cout <<"------------------------" << endl;
    cout << " Each Input(trunc.): Comm. Sent (MiB): " << (Comm_trunc_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_trunc_g + Comm_recv_trunc_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_trunc_g) <<" ms"<< endl;     
    cout <<"------------------------" << endl;
    cout << " Each Input(sole conv.): Comm. Sent (MiB): " << (Comm_soleconv_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_soleconv_g + Comm_recv_soleconv_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_soleconv_g) <<" ms"<< endl; 
    cout <<"------------------------" << endl;
    cout << " Each Input(max pooling): Comm. Sent (MiB): " << (Comm_mxp_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_mxp_g + Comm_recv_mxp_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_mxp_g) <<" ms"<< endl; 
    cout <<"------------------------" << endl;
    cout << " Each Input(sole fc): Comm. Sent (MiB): " << (Comm_solefc_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_solefc_g + Comm_recv_solefc_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_solefc_g) <<" ms"<< endl; 
    cout <<"------------------------" << endl;
    cout << " Each Input(argmax): Comm. Sent (MiB): " << (Comm_argmax_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (Comm_argmax_g + Comm_argmax_recv_g) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (Time_argmax_g) <<" ms"<< endl; 
    cout <<"------------------------" << endl;
    cout << " Each Input(reconv): Comm. Sent (MiB): " << (offComm_total_g + onComm_total_g - Comm_relu_g - Comm_trunc_g - Comm_argmax_g - Comm_soleconv_g - Comm_solefc_g - Comm_mxp_g) << endl;
    if(party == SERVER){
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (offComm_total_g + offComm_recv_g + onComm_total_g + onComm_recv_g - (Comm_relu_g + Comm_recv_relu_g) - (Comm_trunc_g + Comm_recv_trunc_g) - (Comm_argmax_g + Comm_argmax_recv_g) - (Comm_soleconv_g + Comm_recv_soleconv_g) - (Comm_solefc_g + Comm_recv_solefc_g) - (Comm_mxp_g + Comm_recv_mxp_g)) << endl;
    }
    cout << "          : Time (l=" << l << "; b=" << b << ") " << (offTime_total_g + onTime_total_g - Time_relu_g - Time_trunc_g - Time_argmax_g - Time_soleconv_g - Time_solefc_g - Time_mxp_g) <<" ms"<< endl; 
    cout <<"========================" << endl;

	return 0;
}




//this function performs MSB computing for each thread
void field_relu_thread(uint64_t* z, uint64_t* x, int lnum_relu) {
    ReLUFieldProtocol<NetIO, uint64_t>* relu_oracle;
    relu_oracle = new ReLUFieldProtocol<NetIO, uint64_t>(party, FIELD, io, l, b, prime_mod, otpack);
    relu_oracle->relu_pregen(z, x, lnum_relu);
    delete relu_oracle;
    relu_oracle = nullptr;
    return;
}


void sole_relu(uint64_t* relu_z, uint64_t* inp_shrs, int number_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_relu, double &onTime_relu, double &onComm_recv_relu){
    ReLUFieldProtocol<NetIO, uint64_t>* relu_oracle;
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new OTPack<NetIO>(io, party, b, l);
    relu_oracle = new ReLUFieldProtocol<NetIO, uint64_t>(party, FIELD, io, l, b, prime_mod, otpack);
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();    
    relu_oracle->relu(relu_z, inp_shrs, number_relu);
    long long t_on = time_from(start_online);
    uint64_t msbcomm_end = io->counter;
    cout << "relu: Comm. Sent at online (MiB): " << (msbcomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"relu: Online Time (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);
    onComm_relu += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_relu += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (msbcomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        onComm_recv_relu += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed for MSB at online (MiB): " << (msbcomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the relu output
    #if defined(DEBUG_EXEC)
        switch (party) {
            case sci::ALICE: {
                io->send_data(inp_shrs, sizeof(uint64_t)*number_relu);
                io->send_data(relu_z, sizeof(uint64_t)*number_relu);
                break;
            }
            case sci::BOB: {
                uint64_t *xi = new uint64_t[number_relu];
                uint64_t *zi = new uint64_t[number_relu];
                io->recv_data(xi, sizeof(uint64_t)*number_relu);
                io->recv_data(zi, sizeof(uint64_t)*number_relu);
                for(int i=0; i<number_relu; i++){
                    xi[i] = (xi[i] + inp_shrs[i]) % prime_mod;
                    zi[i] = (zi[i] + relu_z[i]) % prime_mod;
                    assert((zi[i] == ((xi[i] <= prime_mod/2) * xi[i]))
                            && "ReLU protocol's answer is incorrect!");
                }
                cout << GREEN << "[Client] Successful ReLU Computing" << RESET << endl;
                delete[] xi;
                delete[] zi;
                break;
            }
        }
    #endif
    delete relu_oracle;
    relu_oracle = nullptr;
    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;    

    return;
}

void sole_div(int div_num, int divisor, int div_const, uint64_t* avrg_pool_l, uint64_t* outp_final_l, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_trunc, double &onTime_trunc, double &onComm_recv_trunc){
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new OTPack<NetIO>(io, party, b, l);
    auto start_div = clock_start();
    //reshape the outArr
	iknpOT = new sci::IKNP<sci::NetIO>(io);
	iknpOTRoleReversed = new sci::IKNP<sci::NetIO>(io); //TCP is full duplex -- so both side OT on same TCP should be good
	kkot = new sci::KKOT<sci::NetIO>(io);
	prg128Instance = new sci::PRG128();
    reluImpl = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack);
	if (party==sci::ALICE){
	iknpOT->setup_send();
	iknpOTRoleReversed->setup_recv();
	}
	else if (party==sci::BOB){
	iknpOT->setup_recv();
	iknpOTRoleReversed->setup_send();
	}
    uint64_t divcomm_start = io->counter;//in bytes
    assert(div_num%8==0 && "number of div not multiples of 8!");
	funcFieldDiv<intType>(party, io, otpack, iknpOT, kkot, reluImpl, prg128Instance, div_num, avrg_pool_l, outp_final_l, divisor, nullptr);
    long long div_end = time_from(start_div);
	uint64_t divcomm_end = io->counter;//in bytes
    cout << "Div: Comm. Sent at online (MiB): " << (divcomm_end - divcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"Div: Online Time (l=" << l << "; b=" << b << ") " << div_end * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (div_end * 1.0 / 1000);  
    onComm_trunc += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    onTime_trunc += (div_end * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (divcomm_end - divcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        onComm_recv_trunc += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Div: Comm. Sent & Recv-ed online (MiB): " << (divcomm_end - divcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the truncation output
    #if defined(DEBUG_EXEC)
        funcTruncationIdeal(div_num, avrg_pool_l, div_const);
        bool anythingFailed = false;
        for(int i=0;i<div_num;i++){
            auto ans = funcReconstruct2PCCons(avrg_pool_l[i],2);
            auto ansigot = funcReconstruct2PCCons(outp_final_l[i],2);
            if (party==sci::BOB){
                if(ans!=ansigot){
                    cout<<RED<<"Error "<<i<<" ans, ansigot = "<<ans<<" "<<ansigot << RESET<<endl;
                    anythingFailed = true;
                }
            }
        }
        if (party == sci::BOB){
            if(anythingFailed == false){    
                cout<<GREEN<<"[Client] right div!"<<RESET<<endl;
            }
        }
    #endif
    delete iknpOT;
    iknpOT = nullptr;
    delete iknpOTRoleReversed;
    iknpOTRoleReversed = nullptr;
    delete kkot;
    kkot = nullptr;
    delete prg128Instance;
    prg128Instance = nullptr;
    delete reluImpl;
    reluImpl = nullptr;

    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;
    /********************************************/
    return;
}


void first_conv_out(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, string input_addr, string wts_addr, string bs_addr, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_soleconv, double &Time_soleconv, double &Comm_soleconv_recv, int SC_inp = 12, int SC_wts = 12, int N = 1){
    int newH = 1 + (image_h+pad_l+pad_r-filter_h)/stride;
    int W = image_h;
    int FW = filter_h;
    int zPadWLeft = pad_l;
    int zPadWRight = pad_r;
    int strideW = stride;
    int newW = newH;
    int CI = inp_chans;
    int CO = out_chans;
    Filters myFilters_mod(CO); 
    Filters myFilters_pt(CO);
    vector<vector<uint64_t>> Kr0result(CO);
    outArr.resize(CO, vector<uint64_t>(newH * newW, 0ULL));
    std::vector<uint64_t> inp_data;
    std::vector<uint64_t> wts;
    std::vector<uint64_t> bs; 
    uint64_t *r0;
    Image imageR0;
    Image imageInp; 
    if(party == CLIENT){
        //path from where C++ executable is running
        std::string filename = input_addr;
        //the vector is flattened from [c, h, w]
        readBinaryFile(filename, inp_data);

    }else{//the server
        // Path to the conv_497 weight file
        std::string binFilePath = wts_addr;
        std::string binBiasPath = bs_addr;
        try {
            // Read the binary file to a vector
            wts = readBinFileToVector(binFilePath);
            bs = readBinFileToVector(binBiasPath);

        } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return;
        }
    }
    /********************************************/
    
    /******* Prepare io, HE and meta-data********/
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    int slot_count =  min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, next_pow2(newH*newH)));
    vector<vector<uint64_t>> shr12off(CO, vector<uint64_t>(slot_count, 0ULL));
    shared_ptr<SEALContext> context_;
    Encryptor* encryptor_;
    Decryptor* decryptor_;
    Evaluator* evaluator_;
    BatchEncoder* encoder_;
    GaloisKeys* gal_keys_;
    Ciphertext* zero_;
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_); 
    uint64_t msbcomm_end;
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
    /********************************************/

    /***************offline computation**********/
    uint64_t offcomm_start = io->counter;
    //begin offline process
    auto start_offline = clock_start();
    if(party == SERVER){
        //reshape the kernel
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
                        int id_temp = out_c*CI*filter_h*FW + inp_c*filter_h*FW + row*FW + col;
                        tmp_chan(row, col) = neg_mod((int64_t)wts[id_temp], prime_mod);
                        k_i2c(out_c, (inp_c*filter_h*FW+row*FW+col)) = neg_mod((int64_t)wts[id_temp], prime_mod);
                        tmp_chan1(row, col) = (int64_t)wts[id_temp];
                    }
                }
                tmp_img[inp_c] = tmp_chan;
                tmp_img1[inp_c] = tmp_chan1;
            }
            myFilters_mod[out_c] = tmp_img;
            myFilters_pt[out_c] = tmp_img1;
        }    
        //recieve r0hat
        const int col_heightR0 = generalData.filter_h * generalData.filter_w * generalData.inp_chans;
        const int col_widthR0 = generalData.output_h * generalData.output_w;
        const int f_size = generalData.filter_h * generalData.filter_w;
        int chanPerCipher = generalData.chans_per_cipher;

        int r0hat_ctNum = ceil(1.0 * col_heightR0 / 1); //one channel for one input
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        }
        //rot-free HE computing with plaintext kernel
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
        shared_ptr<const SEALContext::ContextData> context_data = context_->get_context_data(parms_id);       
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
        
        //reset the shr12off, for chanPerCipher inputs
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < (chanPerCipher * col_widthR0); j++){
                shr12off[i][j] = neg_mod((int64_t)(prime_mod - shr12off[i][j]), prime_mod);
            }
        }
        //gets shr1 for an input
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_widthR0; j++){
                outArr[i][j] = shr12off[i][j];
            }
        }

        if(verbose_info){
            cout << "[Server] offline share generated" << endl;
        }        
    }else{//the client
        //generate r0
        int num_relu = CI*image_h*W;
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
        const int col_widthR0 = generalData.output_h * generalData.output_w;
        Channel image_colR0(col_heightR0, col_widthR0);
        i2c(p_imageR0, image_colR0, generalData.filter_h, generalData.filter_w, generalData.stride_h, generalData.stride_w, generalData.output_h, generalData.output_w);     
        //encrypt r0hat
        int chanPerCipher = generalData.chans_per_cipher;
        int cipherNum = ceil(1.0 * col_heightR0 / 1); //one cipher has one output channel to simulate chanPerCipher inputs for batch computation
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
        //form the final share to the output
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_widthR0; j++){
                outArr[i][j] = Kr0result[i][j];
            }
        }
        if(verbose_info){
            cout << "[Client] output share formed" << endl;
        }        
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "Conv: Comm. Sent at offline per input (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*generalData.chans_per_cipher) << endl;
    cout <<"Conv: Offline Time per input (l=" << l << "; b=" << b << ") " << t_off * 1.0 / (1000*generalData.chans_per_cipher) <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*generalData.chans_per_cipher));
    offTime_total += (t_off * 1.0 / (1000*generalData.chans_per_cipher));  
    Comm_soleconv += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*generalData.chans_per_cipher));
    Time_soleconv += (t_off * 1.0 / (1000*generalData.chans_per_cipher));  
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)*generalData.chans_per_cipher));
        Comm_soleconv_recv += (myRecev / (1.0*(1ULL << 20)*generalData.chans_per_cipher));
        //cout << "Conv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)*generalData.chans_per_cipher) << endl;
    }
    //io->flush();
    //delete io;
    //io = nullptr;
    /********************************************/

    /**************online computation************/
    //io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();
    if(party == CLIENT){
        imageInp.resize(CI); 
        #pragma omp parallel for num_threads(numThreads) schedule(static)        
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod(((int64_t)inp_data[idx]-(int64_t)r0[idx]), (int64_t)prime_mod);
                }
            }
            imageInp[chan] = tmp_chan;
        }        
        //send the masked input to server
        for(int i = 0; i < CI; i++) {
            io->send_data(imageInp[i].data(), image_h * W * sizeof(uint64_t));
        }
        if(verbose_info){
            cout << "[Client] masked input sent" << endl;
        }
    }else{//the server
        //recieve the masked input
        Image image_in(CI);
        for(int i = 0; i < CI; i++) {
            image_in[i].resize(image_h, W);
            io->recv_data(image_in[i].data(), image_h * W * sizeof(uint64_t));
        }
        //perform the convolution
        //the filter values should be small enough to fit uint64_t
        Image local_xr0k = ideal_function(image_in, myFilters_pt, generalData);
        //add the bias and noise share
        const int col_widthR0 = generalData.output_h * generalData.output_w;
        int chanPerCipher = generalData.chans_per_cipher;
        for(int i = 0; i < CO; i++){
            uint64_t bs_temp = neg_mod((int64_t)bs[i],prime_mod) * pow(2,SC_inp);
            for(int j = 0; j < col_widthR0; j++){
                int row_tmp = j / generalData.output_w;
                int col_tmp = j % generalData.output_w;
                uint64_t lo_temp = neg_mod((int64_t)local_xr0k[i](row_tmp, col_tmp), prime_mod);
                outArr[i][j] = (outArr[i][j] + lo_temp) % prime_mod;
                outArr[i][j] = (outArr[i][j] + bs_temp) % prime_mod;
            }
        }
        if(verbose_info){
            cout << "[Server] output share formed" << endl;
        }
    }
    long long t_on = time_from(start_online);
    uint64_t oncomm_end = io->counter;
    cout << "Conv: Comm. Sent at online (MiB): " << (oncomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"Conv: Online Time (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);  
    Comm_soleconv += (double(oncomm_end - oncomm_start)/(1.0*(1ULL << 20)));
    Time_soleconv += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (oncomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_soleconv_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Conv: Comm. Sent & Recv-ed online (MiB): " << (oncomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    /********************************************/

    /***************verification*****************/
    #if defined(DEBUG_EXEC)
        if(party == CLIENT){
            //send final share
            for(int i = 0; i < CO; i++) {
                io->send_data(outArr[i].data(), sizeof(uint64_t) * (generalData.output_h * generalData.output_w));
            } 
            //form the input image
            Image imageInp(CI);
            for (int chan = 0; chan < CI; chan++) {
                Channel tmp_chan(image_h, W);
                for (int h = 0; h < image_h; h++) {
                    for (int w = 0; w < W; w++) {
                        int idx = chan * image_h * W + h * W + w;
                        tmp_chan(h, w) = neg_mod((int64_t)inp_data[idx], prime_mod);
                    }
                }
                imageInp[chan] = tmp_chan;
            }
            //send input data
            for(int i = 0; i < CI; i++) {
                io->send_data(imageInp[i].data(), image_h * W * sizeof(uint64_t));
            }
        }else{//the server
            //receive final share
            vector<vector<uint64_t>> outArr_0;
            outArr_0.resize(CO);
            int col_w = generalData.output_h * generalData.output_w;
            for(int i = 0; i < CO; i++) {
                outArr_0[i].resize(col_w);
                io->recv_data(outArr_0[i].data(), sizeof(uint64_t) * col_w);
            }
            //get the result from final shares
            for(int i = 0; i < CO; i++) {
                for(int j = 0; j < col_w; j++) {
                    outArr_0[i][j] = (outArr_0[i][j] + outArr[i][j]) % prime_mod;
                }
            }
            //receive input
            Image image_in(CI);
            for(int i = 0; i < CI; i++) {
                image_in[i].resize(image_h, W);
                io->recv_data(image_in[i].data(), image_h * W * sizeof(uint64_t));
            }             
            //get the convolution
            Image resultConv = ideal_function(image_in, myFilters_pt, generalData);
            //compare the result
            bool pass = true;
            for (int i = 0; i < CO; i++) {
                uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2,SC_wts);
                for (int j = 0; j < newH; j++) {
                    for (int k = 0; k < newW; k++) {
                        int idx = j * newW + k;
                        resultConv[i](j,k) = (neg_mod(resultConv[i](j,k),(int64_t)prime_mod) + bs_temp) % prime_mod;
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
                cout << RED << "WARNING: The implementation assumes that the computation performed by the server (on it's model and h5)" << endl;
                cout << "fits in a 64-bit integer. The failed operation could be a result of overflowing the bound." << RESET << endl;
            }
        }
    #endif
    /********************************************/

    /********clear vector except the outArr******/
    //clear vector by assigning std::vector<int>();
    if (party == CLIENT){
        delete[] r0; 
        r0 = nullptr;
    }
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
    context_ = nullptr;
    encryptor_ = nullptr;
    decryptor_ = nullptr;
    evaluator_ = nullptr;
    encoder_ = nullptr;
    gal_keys_ = nullptr;
    zero_ = nullptr;   
    
    io->flush();
    delete io;
    io = nullptr;
    /********************************************/
    return;
}

void mxp(uint64_t *x, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_mxp, double &onTime_mxp, double &onComm_recv_mxp){
    /***************prepare the data*************/
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    int mxprows_each = generalData.output_h*generalData.output_w;
    int mxpcols_each = generalData.filter_h*generalData.filter_w;
    otpack = new sci::OTPack<sci::NetIO>(io, party, baseForRelu, bitlength);
    /********************************************/

    /****************maxpool*********************/
    MaxPoolProtocol<NetIO, uint64_t>* maxpool_oracle;
    maxpool_oracle = new MaxPoolProtocol<NetIO, uint64_t>(party, FIELD, io, l, b, prime_mod, otpack);
    auto start_mxp = clock_start();
    uint64_t mxpcomm_start = io->counter;//in bytes
    for(int i = 0; i < generalData.out_chans; i++){
        uint64_t *x_one = new uint64_t[mxprows_each*mxpcols_each];
        Image oneImg(1);
        Channel tmp_chan(generalData.image_h, generalData.image_w);
        int chan_offset = i * generalData.image_h*generalData.image_w;
        for(int j = 0; j < (generalData.image_h*generalData.image_w); j++){
            int row_tmp = j / generalData.image_w;
            int col_tmp = j % generalData.image_w;
            tmp_chan(row_tmp, col_tmp) = outArr[i][j];

        }
        oneImg[0] = tmp_chan;
        auto p_img = pad_image(generalData, oneImg);
        Channel img_col((generalData.filter_h*generalData.filter_w), (generalData.output_h*generalData.output_w));
        i2c(p_img, img_col, generalData.filter_h, generalData.filter_w, generalData.stride_h, generalData.stride_w, generalData.output_h, generalData.output_w);
        for(int j = 0; j < (generalData.output_h*generalData.output_w); j++){
            for(int k = 0; k < (generalData.filter_h*generalData.filter_w); k++){
                x_one[j*(generalData.filter_h*generalData.filter_w) + k] = img_col(k, j);
            }
        }
        maxpool_oracle->funcMaxMPC(mxprows_each, mxpcols_each, x_one, x+(i*mxprows_each), nullptr);
        delete[] x_one;
        x_one = nullptr;
    }
    delete maxpool_oracle;
    maxpool_oracle = nullptr;
    long long t_mxp = time_from(start_mxp);
    uint64_t mxpcomm_end = io->counter;//in bytes
    cout << "MaxPool: Comm. Sent at online (MiB): " << (mxpcomm_end - mxpcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"MaxPool: Online Time (l=" << l << "; b=" << b << ") " << t_mxp * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(mxpcomm_end - mxpcomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_mxp * 1.0 / 1000);
    onComm_mxp += (double(mxpcomm_end - mxpcomm_start)/(1.0*(1ULL<<20)));
    onTime_mxp += (t_mxp * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (mxpcomm_end - mxpcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        onComm_recv_mxp += (myRecev / (1.0*(1ULL << 20)));
        //cout << "MaxPool: Comm. Sent & Recv-ed online (MiB): " << (mxpcomm_end - mxpcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    /********************************************/
    
    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;
    return;
}

void trunc_func(int div_num, int consSF, uint64_t *outp_final, uint64_t *x, double &onComm_total, double &onTime_total, double &onComm_recv, double &onComm_trunc, double &onTime_trunc, double &onComm_recv_trunc){
    uint32_t divisor = 1<<consSF;
    uint64_t *outp = new uint64_t[div_num];
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new sci::OTPack<sci::NetIO>(io, party, baseForRelu, bitlength);
    //pregenerate the share at server
    if(party == SERVER){
        prg.random_mod_p<uint64_t>(outp_final, div_num, prime_mod);  
    } 
    auto start_div = clock_start();
	iknpOT = new sci::IKNP<sci::NetIO>(io);
	iknpOTRoleReversed = new sci::IKNP<sci::NetIO>(io); //TCP is full duplex -- so both side OT on same TCP should be good
	kkot = new sci::KKOT<sci::NetIO>(io);
	prg128Instance = new sci::PRG128();
    reluImpl = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack);
	if (party==sci::ALICE){
	iknpOT->setup_send();
	iknpOTRoleReversed->setup_recv();
	}
	else if (party==sci::BOB){
	iknpOT->setup_recv();
	iknpOTRoleReversed->setup_send();
	}
    uint64_t divcomm_start = io->counter;//in bytes
    assert(div_num%8==0 && "number of div not multiples of 8!");
	funcFieldDiv<intType>(party, io, otpack, iknpOT, kkot, reluImpl, prg128Instance, div_num, x, outp, divisor, nullptr);
    //form the final share
    if(party == SERVER){
        for(int i = 0; i < div_num; i++){
            outp[i] = (outp[i] + outp_final[i]) % prime_mod;
            outp_final[i] = neg_mod((int64_t)(prime_mod - outp_final[i]), prime_mod);
        }
        io->send_data(outp, sizeof(uint64_t)*div_num);
        if(verbose_info){
            cout << "[Server] div share sent" << endl;
        }
    }else{//the client
        io->recv_data(outp_final, sizeof(uint64_t)*div_num);
        for(int i = 0; i < div_num; i++){
            outp_final[i] = (outp[i] + outp_final[i]) % prime_mod;
        }
        if(verbose_info){
            cout << "[Client] div share formed" << endl;

        }
    }
    long long div_end = time_from(start_div);
	uint64_t divcomm_end = io->counter;//in bytes
    cout << "Div: Comm. Sent at online (MiB): " << (divcomm_end - divcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"Div: Online Time (l=" << l << "; b=" << b << ") " << div_end * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (div_end * 1.0 / 1000);  
    onComm_trunc += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    onTime_trunc += (div_end * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (divcomm_end - divcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        onComm_recv_trunc += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Div: Comm. Sent & Recv-ed online (MiB): " << (divcomm_end - divcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the truncation output
    #if defined(DEBUG_EXEC)
        funcTruncationIdeal(div_num, x, consSF);
        bool anythingFailed = false;
        for(int i=0;i<div_num;i++){
            auto ans = funcReconstruct2PCCons(x[i],2);
            auto ansigot = funcReconstruct2PCCons(outp_final[i],2);
            if (party==sci::BOB){
                if(ans!=ansigot){
                    cout<<RED<<"Error "<<i<<" ans, ansigot = "<<ans<<" "<<ansigot << RESET<<endl;
                    anythingFailed = true;
                }
            }
        }
        if (party == sci::BOB){
            cout<<GREEN<<"[Client] right div!"<<RESET<<endl;
        }
    #endif
    delete iknpOT;
    iknpOT = nullptr;
    delete iknpOTRoleReversed;
    iknpOTRoleReversed = nullptr;
    delete kkot;
    kkot = nullptr;
    delete prg128Instance;
    prg128Instance = nullptr;
    delete reluImpl;
    reluImpl = nullptr;
    delete otpack;
    otpack = nullptr;
    delete[] outp;
    outp = nullptr;
    io->flush();
    delete io;
    io = nullptr;
    return;
}


void reconv_normal(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, vector<vector<uint64_t>> &outArr_two, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1){
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new OTPack<NetIO>(io, party, b, l);
    int newH = 1 + (image_h+pad_l+pad_r-filter_h)/stride;
    int W = image_h;
    int FW = filter_h;
    int zPadWLeft = pad_l;
    int zPadWRight = pad_r;
    int strideW = stride;
    int newW = newH;
    int CI = inp_chans;
    int CO = out_chans;
    Filters myFilters_mod(CO); 
    Filters myFilters_pt(CO);
    vector<vector<uint64_t>> Kr0result(CO);
    std::vector<uint64_t> wts;
    std::vector<uint64_t> bs;
    outArr_two.resize(CO, vector<uint64_t>(newH * newW, 0ULL));
    if(party == SERVER){//the server
        // Path to the conv weight file
        std::string binFilePath = wts_addr;
        std::string binBiasPath = bs_addr;
        try {
            // Read the binary file to a vector
            wts = readBinFileToVector(binFilePath);
            bs = readBinFileToVector(binBiasPath);
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return;
        }
    }
    /********************************************/

    /******* Prepare io, HE and meta-data********/
    int slot_count =  min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, next_pow2(newH*newH)));
    vector<vector<uint64_t>> shr12off(CO, vector<uint64_t>(slot_count, 0ULL));
    shared_ptr<SEALContext> context_;
    Encryptor* encryptor_;
    Decryptor* decryptor_;
    Evaluator* evaluator_;
    BatchEncoder* encoder_;
    GaloisKeys* gal_keys_;
    Ciphertext* zero_;    
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
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
    int number_relu = CI * image_h * W;
    uint64_t *bool_shr = new uint64_t[number_relu];//boolean share
    int num_ct_g1h3 = ceil(1.0 * number_relu / slot_count);
    vector<Ciphertext> enc_g1h3;
    uint64_t *r0;

    uint64_t offcomm_g1h3 = 0;
    /********************************************/

    /***************reluconv-503 offline*********/
    uint64_t offcomm_start = io->counter;
    auto start_offline = clock_start();    
    if(party == SERVER){
        //the server generates the share of MSB
        bool *g1 = new bool[number_relu];
        prg.random_bool(g1, number_relu);
        for(int j = 0; j< number_relu; j++){
            bool_shr[j] = g1[j];
        }
        delete[] g1;
        //the server encrypts the g1 and h3
        vector<Ciphertext> g1h3_ct(2 * num_ct_g1h3);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_g1h3; i++){
            vector<uint64_t> v1(slot_count, 0ULL);//it's g1
            vector<uint64_t> v2(slot_count, 0ULL);//it's h3
            Plaintext tmp1, tmp2;
            int idx_offset = i * slot_count;
            if(i == (num_ct_g1h3 - 1)){
                for(int j = 0; j < (number_relu - idx_offset); j++){
                    v1[j] = bool_shr[idx_offset + j];
                    if ((int64_t)bool_shr[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)outp_final[idx_offset + j], prime_mod);
                    }else{
                        v2[j] = neg_mod(-(int64_t)outp_final[idx_offset + j], prime_mod);
                    }
                }
            }else{
                for(int j = 0; j < slot_count; j++){
                    v1[j] = bool_shr[idx_offset + j];
                    if ((int64_t)bool_shr[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)outp_final[idx_offset + j], prime_mod);
                    }else{
                        v2[j] = neg_mod(-(int64_t)outp_final[idx_offset + j], prime_mod);
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
        //reshape the kernel
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
                        int id_temp = out_c*CI*filter_h*FW + inp_c*filter_h*FW + row*FW + col;
                        tmp_chan(row, col) = neg_mod((int64_t)wts[id_temp], prime_mod);
                        k_i2c(out_c, (inp_c*filter_h*FW+row*FW+col)) = neg_mod((int64_t)wts[id_temp], prime_mod);
                        tmp_chan1(row, col) = (int64_t)wts[id_temp];
                    }
                }
                tmp_img[inp_c] = tmp_chan;
                tmp_img1[inp_c] = tmp_chan1;
            }
            myFilters_mod[out_c] = tmp_img;
            myFilters_pt[out_c] = tmp_img1;
        } 

        //recieve r0hat
        const int col_heightR0 = generalData.filter_h * generalData.filter_w * generalData.inp_chans;
        const int col_widthR0 = generalData.output_h * generalData.output_w;
        const int f_size = generalData.filter_h * generalData.filter_w;
        int chanPerCipher = generalData.chans_per_cipher;
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / 1);//one channel for one input
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
        //gets partial share
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_widthR0; j++){
                outArr_two[i][j] = shr12off[i][j];
            }
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }else{//the client
        //generate r0
        r0 = new uint64_t[number_relu];
        prg.random_mod_p<uint64_t>(r0, number_relu, prime_mod);        
        //transform r0 into r0hat and perform encryption
        Image imageR0(CI);
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
        const int col_widthR0 = generalData.output_h * generalData.output_w;
        Channel image_colR0(col_heightR0, col_widthR0);
        i2c(p_imageR0, image_colR0, generalData.filter_h, generalData.filter_w, generalData.stride_h, generalData.stride_w, generalData.output_h, generalData.output_w);     
        //encrypt r0hat
        int chanPerCipher = generalData.chans_per_cipher;
        int cipherNum = ceil(1.0 * col_heightR0 / 1);//one channel for one input
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
        //get shr0
        for(int i = 0; i < CO; i++) {
            for(int j = 0; j < col_widthR0; j++) {
                outArr_two[i][j] = Kr0result[i][j];
            }
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "reconv: Comm. Sent at offline per input (MiB): " << (offcomm_end - offcomm_start - offcomm_g1h3)/(1.0*(1ULL << 20)*generalData.chans_per_cipher) + (offcomm_g1h3/(1.0*(1ULL << 20))) << endl;
    cout <<"reconv: Offline Time per input (l=" << l << "; b=" << b << ") " << t_off * 1.0 / (1000*generalData.chans_per_cipher) <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start - offcomm_g1h3)/(1.0*(1ULL << 20)*generalData.chans_per_cipher) + (offcomm_g1h3/(1.0*(1ULL << 20))));
    offTime_total += (t_off * 1.0 / (1000*generalData.chans_per_cipher));    
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)*generalData.chans_per_cipher));
        //cout << "reconv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start - offcomm_g1h3 + myRecev)/(1.0*(1ULL << 20)*generalData.chans_per_cipher) + (offcomm_g1h3/(1.0*(1ULL << 20))) << endl;
    }    
    /********************************************/

    /***************reluconv-503 online**********/
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();
    field_relu_thread(bool_shr,outp_final,number_relu);
    long long t_on = time_from(start_online);
    uint64_t msbcomm_end = io->counter;
    cout << "reconv: Comm. Sent for MSB at online (MiB): " << (msbcomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Online Time for MSB (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (msbcomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed for MSB at online (MiB): " << (msbcomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the MSB output
    #if defined(DEBUG_EXEC)
        switch (party) {
            case sci::ALICE: {
                io->send_data(outp_final, sizeof(uint64_t) * number_relu);
                io->send_data(bool_shr, sizeof(uint64_t) * number_relu);
                break;
            }
            case sci::BOB: {
                uint64_t *xi = new uint64_t[number_relu];
                uint64_t *zi = new uint64_t[number_relu];
                io->recv_data(xi, sizeof(uint64_t) * number_relu);
                io->recv_data(zi, sizeof(uint64_t) * number_relu);
                for(int i=0; i<number_relu; i++){
                    xi[i] = (xi[i] + outp_final[i]) % prime_mod;
                    zi[i] = (zi[i] + bool_shr[i]) % 2;//this recovers the MSB from two boolean shares
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
    oncomm_start = io->counter;
    start_online = clock_start(); 
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
                for(int j = 0; j < (number_relu - idx_offset); j++){
                    v1[j] = (bool_shr[idx_offset + j] ^ 1);//the drelu share
                    if ((int64_t)bool_shr[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)outp_final[idx_offset + j] - (int64_t)r0[idx_offset + j], prime_mod);//it's h1
                        v3[j] = neg_mod(-(int64_t)outp_final[idx_offset + j], prime_mod);//it's h2
                    }else{
                        v2[j] = neg_mod(-(int64_t)r0[idx_offset + j], prime_mod);
                        v3[j] = neg_mod((int64_t)outp_final[idx_offset + j], prime_mod);
                    }
                }
            }else{
                for(int j = 0; j < slot_count; j++){
                    v1[j] = (bool_shr[idx_offset + j] ^ 1);
                    if ((int64_t)bool_shr[idx_offset + j] == 0){
                        v2[j] = neg_mod((int64_t)outp_final[idx_offset + j] - (int64_t)r0[idx_offset + j], prime_mod);//it's h1
                        v3[j] = neg_mod(-(int64_t)outp_final[idx_offset + j], prime_mod);//it's h2
                        
                    }else{
                        v2[j] = neg_mod(-(int64_t)r0[idx_offset + j], prime_mod);
                        v3[j] = neg_mod((int64_t)outp_final[idx_offset + j], prime_mod);
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
    }else{//the server
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
        
        //form the image
        Image imageH5(CI);
        #pragma omp parallel for num_threads(numThreads) schedule(static)        
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)pt_h5[idx / slot_count][idx % slot_count], prime_mod);
                    if((int64_t)bool_shr[idx] == 1){
                        uint64_t tp = neg_mod((int64_t)outp_final[idx], prime_mod);
                        tmp_chan(h, w) = (tmp_chan(h, w) + tp) % prime_mod;
                        tmp_chan(h, w) = neg_mod((int64_t)tmp_chan(h, w), prime_mod);
                    }

                }
            }
            imageH5[chan] = tmp_chan;
        }


        //do the convolution and the filter values should be small enough to fit uint64_t
        Image local_kH5 = ideal_function(imageH5, myFilters_pt, generalData);
        //form the final share
        const int col_w = generalData.output_h * generalData.output_w;
        for(int i = 0; i < CO; i++){
            uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2,SC_wts);
            for(int j = 0; j < col_w; j++){
                outArr_two[i][j] = (outArr_two[i][j]+neg_mod((int64_t)local_kH5[i](j / generalData.output_w, j % generalData.output_w), prime_mod)) % prime_mod;
                outArr_two[i][j] = (outArr_two[i][j]+bs_temp)% prime_mod; 
                outArr_two[i][j] = neg_mod((int64_t)outArr_two[i][j], prime_mod);
            }
        }
        if(verbose_info){cout << "[Server] output share formed" << endl;}
    }
    t_on = time_from(start_online);
    uint64_t oncomm_end = io->counter;
    cout << "reconv: Comm. Sent after MSB at online (MiB): " << (oncomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Online Time after MSB (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
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
        //cout << "reconv: Comm. Sent & Recv-ed online after MSB (MiB): " << (oncomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the output
    #if defined(DEBUG_EXEC)
        switch (party) {
            case sci::BOB: {
                //form the input image
                Image imageInp(CI);
                for (int chan = 0; chan < CI; chan++) {
                    Channel tmp_chan(image_h, W);
                    for (int h = 0; h < image_h; h++) {
                        for (int w = 0; w < W; w++) {
                            int idx = chan * image_h * W + h * W + w;
                            tmp_chan(h, w) = neg_mod((int64_t)outp_final[idx], prime_mod);
                        }
                    }
                    imageInp[chan] = tmp_chan;
                }
                //send input share
                for(int i = 0; i < CI; i++) {
                    io->send_data(imageInp[i].data(), image_h * W * sizeof(uint64_t));
                }
                //send MSB share
                io->send_data(bool_shr, sizeof(uint64_t) * number_relu); 
                const int col_w = generalData.output_h * generalData.output_w;
                //send final share
                for(int i = 0; i < CO; i++) {
                    io->send_data(outArr_two[i].data(), sizeof(uint64_t) * (col_w));
                }                  
                break;
            }
            case sci::ALICE: {
                //receive input share
                Image image_in(CI);
                for(int i = 0; i < CI; i++) {
                    image_in[i].resize(image_h, W);
                    io->recv_data(image_in[i].data(), image_h * W * sizeof(uint64_t));
                }   
                //receive MSB share
                uint64_t *zi = new uint64_t[number_relu];
                io->recv_data(zi, sizeof(uint64_t) * number_relu);
                //form the input image
                for(int i = 0; i < CI; i++) {
                    for(int h = 0; h < image_h; h++) {
                        for(int w = 0; w < W; w++) {
                            int idx = i * image_h * W + h * W + w;
                            image_in[i](h,w) = (neg_mod((int64_t)outp_final[idx], prime_mod) + image_in[i](h,w)) % prime_mod;
                            int drelu_tmp = (bool_shr[idx] + zi[idx] + 1) % 2;
                            image_in[i](h,w) = image_in[i](h,w) * drelu_tmp;
                        }
                    }
                }
                //get the convolution
                Image resultConv = ideal_function(image_in, myFilters_pt, generalData);
                //receive final share
                vector<vector<uint64_t>> outArr_0;
                outArr_0.resize(CO);
                const int col_w = generalData.output_h * generalData.output_w;
                for(int i = 0; i < CO; i++) {
                    outArr_0[i].resize(col_w);
                    io->recv_data(outArr_0[i].data(), sizeof(uint64_t) * col_w);
                }
                //get the result from final shares
                for(int i = 0; i < CO; i++) {
                    for(int j = 0; j < col_w; j++) {
                        outArr_0[i][j] = (outArr_0[i][j] + outArr_two[i][j]) % prime_mod;
                    }
                }
                //compare the result
                bool pass = true;
                for (int i = 0; i < CO; i++) {
                    uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2,SC_wts);
                    for (int j = 0; j < newH; j++) {
                        for (int k = 0; k < newW; k++) {
                            int idx = j * newW + k;
                            resultConv[i](j,k) = (neg_mod(resultConv[i](j,k),(int64_t)prime_mod) + bs_temp) % prime_mod;
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
                    cout << RED << "WARNING: The implementation assumes that the computation performed by the server (on it's model and h5)" << endl;
                    cout << "fits in a 64-bit integer. The failed operation could be a result of overflowing the bound." << RESET << endl;
                }
                delete[] zi;                
                break;
            }
        }
    #endif
    //clear the memory except outArr
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
    context_ = nullptr;
    encryptor_ = nullptr;
    decryptor_ = nullptr;
    evaluator_ = nullptr;
    encoder_ = nullptr;
    gal_keys_ = nullptr;
    zero_ = nullptr;
    if(party == CLIENT){
        delete[] r0;
        r0 = nullptr;
    } 
    delete[] bool_shr;
    bool_shr = nullptr;

    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;
    
    /********************************************/    
    return;
}

void final_fc_max(uint64_t *outp_final, int x_scales, int inp_dim, int out_dim, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_solefc, double &Time_solefc, double &Comm_solefc_recv, double &Comm_argmax, double &Time_argmax, double &Comm_argmax_recv){
    //offline computation
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new OTPack<NetIO>(io, party, b, l);

    int slot_count =  min(SEAL_POLY_MOD_DEGREE_MAX, 8192);
    shared_ptr<SEALContext> context_;
    Encryptor* encryptor_;
    Decryptor* decryptor_;
    Evaluator* evaluator_;
    BatchEncoder* encoder_;
    GaloisKeys* gal_keys_;
    Ciphertext* zero_; 
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);

    std::vector<uint64_t> wts;
    std::vector<uint64_t> bs;

    if(party == SERVER){//the server
        // Path to the conv weight file
        std::string binFilePath = wts_addr;
        std::string binBiasPath = bs_addr;
        try {
            // Read the binary file to a vector
            wts = readBinFileToVector(binFilePath);
            bs = readBinFileToVector(binBiasPath);
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return;
        }
    }
    uint64_t offcomm_start = io->counter;
    auto start_offline = clock_start();
    uint64_t *r0;
    uint64_t *outFC = new uint64_t[out_dim];
    vector<vector<uint64_t>> noise_fc(out_dim, vector<uint64_t>(slot_count, 0ULL));//for server
    Channel chan_mod(out_dim, inp_dim);
    Channel chan_pt(out_dim, inp_dim);
    if(party == CLIENT){
        r0 = new uint64_t[inp_dim];
        //generate the r0
        prg.random_mod_p<uint64_t>(r0, inp_dim, prime_mod);
        int num_per_cipher = slot_count / 1;
        int num_ct_r0 = ceil(1.0*inp_dim / 1);//one element for one input
        vector<Ciphertext> enc_r0(num_ct_r0);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_r0; i++){
            vector<uint64_t> v1(slot_count, 0ULL);
            Plaintext tmp;
            v1[0] = r0[i];

            encoder_->encode(v1, tmp);
            encryptor_->encrypt(tmp, enc_r0[i]);
            evaluator_->mod_switch_to_next_inplace(enc_r0[i]);
        }
        //send the r0
        send_encrypted_vector(io, enc_r0);
        if(verbose_info){
            cout << "[Client] encrypted r0 sent" << endl;
        }
        //recieve the masked kR0
        vector<Ciphertext> ct_Kr0(out_dim);
        recv_encrypted_vector(io, ct_Kr0);
        if(verbose_info){
            cout << "[Client] masked Kr0 received" << endl;
        }

        //decrypt the masked kR0
        Plaintext tmp;
        vector<uint64_t> kr0_out(slot_count, 0ULL);
        for(int ct_idx = 0; ct_idx < out_dim; ct_idx++) {
            vector<uint64_t> kr0_out_tp(slot_count, 0ULL);
            decryptor_->decrypt(ct_Kr0[ct_idx], tmp);
            encoder_->decode(tmp, kr0_out_tp);
            kr0_out[ct_idx] = kr0_out_tp[0];
        }

        if(verbose_info){cout << "[Client] share decrypted" << endl;}
        
        //get the kr0 share
        for(int i = 0; i < out_dim; i++) {
            outFC[i] = kr0_out[i];
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }else{//the server
        //reshape the weight
        #pragma omp parallel for num_threads(numThreads) schedule(static)         
        for (int row = 0; row < out_dim; row++) {
            for (int col = 0; col < inp_dim; col++) {
                int id_temp = row*inp_dim + col;
                chan_mod(row, col) = neg_mod((int64_t)wts[id_temp], prime_mod);
                chan_pt(row, col) = (int64_t)wts[id_temp];
            }
        }   
        int num_per_cipher = slot_count / 1;
        int num_ct_r0 = ceil(1.0*inp_dim / 1);//one element for one input
        vector<Ciphertext> enc_r0hat(num_ct_r0);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0hat received" << endl;
        }
        //perform rot-free computation
        vector<Ciphertext> enc_Kr0(out_dim);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < inp_dim; i++){       
            //get one col in 2-dim kernel
            vector<uint64_t> k_row(out_dim, 0ULL);
            vector<Ciphertext> tp_Kr0(out_dim);
            int nonzero_num = 0;
            Ciphertext const_cipher = enc_r0hat[i];
            for(int j = 0; j < out_dim; j++){
                k_row[j] = chan_mod(j, i);
                if(chan_mod(j, i) != 0)nonzero_num++;
                if(i == 0){
                    enc_Kr0[j] = *zero_;
                    evaluator_->mod_switch_to_next_inplace(enc_Kr0[j]);
                }
            }
                        
            //get the mult and add information
            MultiplierOptimizer optimizer(out_dim);
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
        
        //add the noise
        for(int i = 0; i < out_dim; i++){
            prg.random_mod_p<uint64_t>(noise_fc[i].data(), slot_count, prime_mod);
            Plaintext tmp_res;
            encoder_->encode(noise_fc[i], tmp_res);
            evaluator_->add_plain_inplace(enc_Kr0[i], tmp_res);
        }

        //perform the noise flooding
        parms_id_type parms_id = enc_Kr0[0].parms_id();
        shared_ptr<const SEALContext::ContextData> context_data
        = context_->get_context_data(parms_id);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int ct_idx = 0; ct_idx < out_dim; ct_idx++) {
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
        //reset the noise
        for(int i = 0; i < out_dim; i++){
            outFC[i] = neg_mod((int64_t)(prime_mod - noise_fc[i][0]), prime_mod);
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "fc: Comm. Sent at offline per input (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count) << endl;
    cout <<"fc: Offline Time per input (l=" << l << "; b=" << b << ") " << t_off * 1.0 / (1000*slot_count) <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count));
    offTime_total += (t_off * 1.0 / (1000*slot_count));    
    Comm_solefc += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count));
    Time_solefc += (t_off * 1.0 / (1000*slot_count));
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)*slot_count));
        Comm_solefc_recv += (myRecev / (1.0*(1ULL << 20)*slot_count));
        //cout << "fc: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)*slot_count) << endl;
    } 
    //online computation
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();
    if(party == CLIENT){
        //send the masked input
        Channel mask_inp(inp_dim, 1);
        for(int i = 0; i < inp_dim; i++){
            mask_inp(i,0) = neg_mod(((int64_t)outp_final[i] - (int64_t)r0[i]), prime_mod);
        }
        //send the masked input to server
        io->send_data(mask_inp.data(), inp_dim * sizeof(uint64_t));
        if(verbose_info){
            cout << "[Client] masked input sent" << endl;
        }
    }else{//the server
        //recieve the masked input
        Channel inp_in(inp_dim,1);
        io->recv_data(inp_in.data(), inp_dim * sizeof(uint64_t));
        if(verbose_info){
            cout << "[Server] masked input received" << endl;
        }
        //add the share
        for(int i = 0; i < inp_dim; i++){
            inp_in(i,0) = (inp_in(i,0) + outp_final[i]) % prime_mod;
            inp_in(i,0) = neg_mod((int64_t)inp_in(i,0), prime_mod);
        }
        //perform the dot product
        //the weight values should be small enough to fit uint64_t
        Channel local_xr0k = chan_pt * inp_in;
        //add the bias and noise share
        for(int i = 0; i < out_dim; i++){
            uint64_t bs_temp = neg_mod((int64_t)bs[i],prime_mod) * pow(2, x_scales);
            uint64_t lo_temp = neg_mod((int64_t)local_xr0k(i,0), prime_mod);
            outFC[i] = (outFC[i] + lo_temp) % prime_mod;
            outFC[i] = (outFC[i] + bs_temp) % prime_mod;
            outFC[i] = neg_mod((int64_t)outFC[i], prime_mod);
        }
        if(verbose_info){
            cout << "[Server] output share formed" << endl;
        }
    }
    long long t_on = time_from(start_online);
    uint64_t oncomm_end = io->counter;
    cout << "fc: Comm. Sent at online (MiB): " << (oncomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"fc: Online Time (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);  
    Comm_solefc += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    Time_solefc += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (oncomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_solefc_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "fc: Comm. Sent & Recv-ed online (MiB): " << (oncomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the output
    #if defined(DEBUG_EXEC)
        if(party == CLIENT){
            //send final share
            io->send_data(outFC, sizeof(uint64_t) * out_dim);
            //form the input image
            Channel imageInp(inp_dim, 1);
            for (int i = 0; i < inp_dim; i++) {
                imageInp(i, 0) = neg_mod((int64_t)outp_final[i], prime_mod);
            }
            //send input data
            io->send_data(imageInp.data(), inp_dim * sizeof(uint64_t));
        }else{//the server
            //receive final share
            vector<uint64_t> outFC_0(out_dim, 0ULL);
            io->recv_data(outFC_0.data(), sizeof(uint64_t) * out_dim);
            //get the result from final shares
            for(int i = 0; i < out_dim; i++) {
                outFC_0[i] = (outFC_0[i] + outFC[i]) % prime_mod;
                outFC_0[i] = neg_mod((int64_t)outFC_0[i], prime_mod);
            }
            //receive input
            Channel image_in(inp_dim,1);
            io->recv_data(image_in.data(), inp_dim * sizeof(uint64_t));  
            //get the input
            for(int i = 0; i < inp_dim; i++){
                image_in(i,0) = (image_in(i,0) + neg_mod((int64_t)outp_final[i], prime_mod)) % prime_mod;
                image_in(i,0) = neg_mod((int64_t)image_in(i,0), prime_mod);
            }           
            //get the dot product
            Channel resultFC = chan_pt * image_in;
            //compare the result
            bool pass = true;
            for (int i = 0; i < out_dim; i++) {
                uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2, x_scales);
                resultFC(i,0) = (neg_mod(resultFC(i,0),(int64_t)prime_mod) + bs_temp) % prime_mod;
                if (outFC_0[i] != neg_mod(resultFC(i,0), (int64_t) prime_mod)){
                    pass = false;
                }
            }
            if (pass) {
                cout << GREEN << "[Server] Successful Online" << RESET << endl;
            }
            else {
                cout << RED << "[Server] Failed Online" << RESET << endl;
                cout << RED << "WARNING: The implementation assumes that the computation performed by the server (on it's model and masked input)" << endl;
                cout << "fits in a 64-bit integer. The failed operation could be a result of overflowing the bound." << RESET << endl;
            }
        }
    #endif
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
    context_ = nullptr;
    encryptor_ = nullptr;
    decryptor_ = nullptr;
    evaluator_ = nullptr;
    encoder_ = nullptr;
    gal_keys_ = nullptr;
    zero_ = nullptr;

    if(party == CLIENT){
        delete[] r0;
        r0 = nullptr;
    }
    
    //get the maximum
	uint64_t* argmax_output_protocol = new uint64_t[1];
	uint64_t* argmax_output_protocol_share_other = new uint64_t[1];
	uint64_t* argmax_output_protocol_arg = new uint64_t[1];
	uint64_t* argmax_output_protocol_share_other_arg = new uint64_t[1];
	uint64_t* argmax_output_actual = new uint64_t[1];
	ArgMaxProtocol<NetIO, uint64_t> argmax_oracle(party, FIELD, io, l, b, prime_mod, otpack);
    switch (party){
        case BOB: {//the client
            uint64_t comm_start = io->counter;
            auto start = clock_start();
            argmax_oracle.ArgMaxMPC(out_dim, outFC, argmax_output_protocol_arg, true, argmax_output_protocol);
            long long t = time_from(start);
            uint64_t comm_end = io->counter;
            cout << "Argmax communication (MiB): " <<BLUE<< ((double)(comm_end - comm_start))/(1.0*(1ULL << 20))<<RESET<< endl;
            cout <<"Argmax Time: "<<GREEN<<(t * 1.0 / 1000)<<" ms" <<RESET<< endl;
            Comm_argmax += ((double)(comm_end - comm_start))/(1.0*(1ULL << 20));
            Time_argmax += (t * 1.0 / 1000);
            std::cout<<"[Client] Done MaxPool protocol execution"<<std::endl;
        
            uint64_t *input_share2 = new uint64_t[out_dim];
            io->recv_data(input_share2, sizeof(uint64_t)*out_dim);
            io->recv_data(argmax_output_protocol_share_other, sizeof(uint64_t)*1);
            io->recv_data(argmax_output_protocol_share_other_arg, sizeof(uint64_t)*1);
            
            //send the communication volume
            uint64_t mySent = (comm_end - comm_start);
            io->send_data(&mySent, sizeof(uint64_t));
            
            cout<<"Checking correctness of ArgMax now..."<<endl;
            argmax_output_protocol[0] = (argmax_output_protocol[0] + argmax_output_protocol_share_other[0]) % prime_mod;
            argmax_output_protocol_arg[0] = (argmax_output_protocol_arg[0] + argmax_output_protocol_share_other_arg[0]) % prime_mod;
            uint64_t max_mag = 0;
            uint64_t max_mag_2 = 0;
            for(int i=0; i<out_dim; i++){
                outFC[i] = (outFC[i] + input_share2[i]) % prime_mod;
                if(outFC[i] < (prime_mod/2)){
                    if(outFC[i] > max_mag){
                        max_mag_2 = max_mag;
                        max_mag = outFC[i];
                    }
                    else if(outFC[i] > max_mag_2){
                        max_mag_2 = outFC[i];
                    }
                }
                else{
                    //the magnitude
                    uint64_t v = prime_mod - outFC[i];
                    if(v > max_mag){
                        max_mag_2 = max_mag;
                        max_mag = v;
                    }
                    else if(v > max_mag_2){
                        max_mag_2 = v;
                    }
                }
            }
            delete[] input_share2;
            input_share2 = nullptr;
            if((max_mag + max_mag) >= (prime_mod/2)){
                cout<<RED<<"Shares exceed their magnitude bound!"<<RESET<<endl;
                    assert(false);
            }
            argmax_output_actual[0] = outFC[0];
            for(int i=1; i<out_dim; i++){
                argmax_output_actual[0] = ((sci::neg_mod(argmax_output_actual[0] - outFC[i], (int64_t)prime_mod)> (prime_mod/2))?outFC[i]:argmax_output_actual[0]);
            }
            std::cout<<"Max Protocol: "<<argmax_output_protocol[0]<<std::endl;
            std::cout<<"Max Actual: "<<argmax_output_actual[0]<<std::endl;
            std::cout<<"ArgMax Protocol: "<<argmax_output_protocol_arg[0]<<std::endl;
            assert(argmax_output_actual[0] == argmax_output_protocol[0] && "ArgMax output is incorrect");
            cout<<"ArgMax answer is: "<<GREEN<<"CORRECT!"<<RESET<<endl;
            break;
        }
        case ALICE: {//the server
            uint64_t comm_start = io->counter;
            auto start_mxp = clock_start();
            argmax_oracle.ArgMaxMPC(out_dim, outFC, argmax_output_protocol_arg, true, argmax_output_protocol);
            long long t_mxp = time_from(start_mxp);
            uint64_t comm_end = io->counter;
            cout << "Argmax communication (MiB): " <<BLUE<< ((double)(comm_end - comm_start))/(1.0*(1ULL << 20)) << RESET<< endl;
            cout <<"Argmax Time: "<<GREEN<< (t_mxp * 1.0 / 1000)<<" ms" <<RESET<< endl;
            Comm_argmax += ((double)(comm_end - comm_start))/(1.0*(1ULL << 20));
            Time_argmax += (t_mxp * 1.0 / 1000);
            std::cout<<"[Server] Done MaxPool protocol execution"<<std::endl;
            io->send_data(outFC, sizeof(uint64_t)*out_dim);
            io->send_data(argmax_output_protocol, sizeof(uint64_t)*1);
            io->send_data(argmax_output_protocol_arg, sizeof(uint64_t)*1);
            
            uint64_t myRecev = 0;
            io->recv_data(&myRecev, sizeof(uint64_t));
            onComm_recv += (myRecev / (1.0*(1ULL << 20)));
            Comm_argmax_recv += (myRecev / (1.0*(1ULL << 20)));
            //cout << "Argmax: Comm. Sent & Recv-ed online (MiB): " << (comm_end - comm_start + myRecev)/(1.0*(1ULL << 20)) << endl;             

            break;
        }
    }    
        
    //clean up
    delete[] argmax_output_protocol;
    argmax_output_protocol = nullptr;
    delete[] argmax_output_protocol_share_other;
    argmax_output_protocol_share_other = nullptr;
    delete[] argmax_output_protocol_arg;
    argmax_output_protocol_arg = nullptr;
    delete[] argmax_output_protocol_share_other_arg;
    argmax_output_protocol_share_other_arg = nullptr;
    delete[] argmax_output_actual;
    argmax_output_actual = nullptr;
    delete[] outFC;
    outFC = nullptr;
    delete otpack;
    otpack = nullptr;
    io->flush();
    delete io;
    io = nullptr;
    
    return;
}

void sole_fc(uint64_t *outp_final, uint64_t *outFC, int col_start, int col_end, int x_scales, int inp_dim, int out_dim, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_solefc, double &Time_solefc, double &Comm_solefc_recv){
    //offline computation
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);

    int slot_count =  min(SEAL_POLY_MOD_DEGREE_MAX, 8192);
    shared_ptr<SEALContext> context_;
    Encryptor* encryptor_;
    Decryptor* decryptor_;
    Evaluator* evaluator_;
    BatchEncoder* encoder_;
    GaloisKeys* gal_keys_;
    Ciphertext* zero_; 
    generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);

    std::vector<uint64_t> wts;
    std::vector<uint64_t> bs;

    if(party == SERVER){//the server
        // Path to the conv weight file
        std::string binFilePath = wts_addr;
        std::string binBiasPath = bs_addr;
        try {
            // Read the binary file to a vector
            wts = readBinFileToVector(binFilePath);
            bs = readBinFileToVector(binBiasPath);
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return;
        }
    }
    uint64_t offcomm_start = io->counter;
    auto start_offline = clock_start();
    uint64_t *r0;
    vector<vector<uint64_t>> noise_fc(out_dim, vector<uint64_t>(slot_count, 0ULL));//for server
    Channel chan_mod(out_dim, (col_end - col_start + 1));
    Channel chan_pt(out_dim, (col_end - col_start + 1));
    if(party == CLIENT){
        r0 = new uint64_t[(col_end - col_start + 1)];
        //generate the r0
        prg.random_mod_p<uint64_t>(r0, col_end - col_start + 1, prime_mod);
        int num_per_cipher = slot_count / 1;
        int num_ct_r0 = ceil(1.0*(col_end - col_start + 1) / 1);//one element for one input
        vector<Ciphertext> enc_r0(num_ct_r0);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_r0; i++){
            vector<uint64_t> v1(slot_count, 0ULL);
            Plaintext tmp;
            v1[0] = r0[i];

            encoder_->encode(v1, tmp);
            encryptor_->encrypt(tmp, enc_r0[i]);
            evaluator_->mod_switch_to_next_inplace(enc_r0[i]);
        }
        //send the r0
        send_encrypted_vector(io, enc_r0);
        if(verbose_info){
            cout << "[Client] encrypted r0 sent" << endl;
        }
        //recieve the masked kR0
        vector<Ciphertext> ct_Kr0(out_dim);
        recv_encrypted_vector(io, ct_Kr0);
        if(verbose_info){
            cout << "[Client] masked Kr0 received" << endl;
        }

        //decrypt the masked kR0
        Plaintext tmp;
        vector<uint64_t> kr0_out(slot_count, 0ULL);
        for(int ct_idx = 0; ct_idx < out_dim; ct_idx++) {
            vector<uint64_t> kr0_out_tp(slot_count, 0ULL);
            decryptor_->decrypt(ct_Kr0[ct_idx], tmp);
            encoder_->decode(tmp, kr0_out_tp);
            kr0_out[ct_idx] = kr0_out_tp[0];
        }

        if(verbose_info){cout << "[Client] share decrypted" << endl;}
        
        //get the kr0 share
        for(int i = 0; i < out_dim; i++) {
            outFC[i] = kr0_out[i];
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }else{//the server
        //reshape the weight
        #pragma omp parallel for num_threads(numThreads) schedule(static)         
        for (int row = 0; row < out_dim; row++) {
            for (int col = col_start; col < (col_end + 1); col++) {
                int id_temp = row*inp_dim + col;
                chan_mod(row, col - col_start) = neg_mod((int64_t)wts[id_temp], prime_mod);
                chan_pt(row, col - col_start) = (int64_t)wts[id_temp];
            }
        }   
        int num_per_cipher = slot_count / 1;
        int num_ct_r0 = ceil(1.0*(col_end - col_start + 1) / 1);//one element for one input
        vector<Ciphertext> enc_r0hat(num_ct_r0);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0hat received" << endl;
        }
        //perform rot-free computation
        vector<Ciphertext> enc_Kr0(out_dim);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < (col_end - col_start + 1); i++){       
            //get one col in 2-dim kernel
            vector<uint64_t> k_row(out_dim, 0ULL);
            vector<Ciphertext> tp_Kr0(out_dim);
            int nonzero_num = 0;
            Ciphertext const_cipher = enc_r0hat[i];
            for(int j = 0; j < out_dim; j++){
                k_row[j] = chan_mod(j, i);
                if(chan_mod(j, i) != 0)nonzero_num++;
                if(i == 0){
                    enc_Kr0[j] = *zero_;
                    evaluator_->mod_switch_to_next_inplace(enc_Kr0[j]);
                }
            }
                        
            //get the mult and add information
            MultiplierOptimizer optimizer(out_dim);
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
        
        //add the noise
        for(int i = 0; i < out_dim; i++){
            prg.random_mod_p<uint64_t>(noise_fc[i].data(), slot_count, prime_mod);
            Plaintext tmp_res;
            encoder_->encode(noise_fc[i], tmp_res);
            evaluator_->add_plain_inplace(enc_Kr0[i], tmp_res);
        }

        //perform the noise flooding
        parms_id_type parms_id = enc_Kr0[0].parms_id();
        shared_ptr<const SEALContext::ContextData> context_data
        = context_->get_context_data(parms_id);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int ct_idx = 0; ct_idx < out_dim; ct_idx++) {
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
        //reset the noise
        for(int i = 0; i < out_dim; i++){
            outFC[i] = neg_mod((int64_t)(prime_mod - noise_fc[i][0]), prime_mod);
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "fc: Comm. Sent at offline per input (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count) << endl;
    cout <<"fc: Offline Time per input (l=" << l << "; b=" << b << ") " << t_off * 1.0 / (1000*slot_count) <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count));
    offTime_total += (t_off * 1.0 / (1000*slot_count));    
    Comm_solefc += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)*slot_count));
    Time_solefc += (t_off * 1.0 / (1000*slot_count));
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)*slot_count));
        Comm_solefc_recv += (myRecev / (1.0*(1ULL << 20)*slot_count));
        //cout << "fc: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)*slot_count) << endl;
    } 
    //online computation
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();
    if(party == CLIENT){
        //send the masked input
        Channel mask_inp((col_end - col_start + 1), 1);
        for(int i = 0; i < (col_end - col_start + 1); i++){
            mask_inp(i,0) = neg_mod(((int64_t)outp_final[i + col_start] - (int64_t)r0[i]), prime_mod);
        }
        //send the masked input to server
        io->send_data(mask_inp.data(), (col_end - col_start + 1) * sizeof(uint64_t));
        if(verbose_info){
            cout << "[Client] masked input sent" << endl;
        }
    }else{//the server
        //recieve the masked input
        Channel inp_in((col_end - col_start + 1),1);
        io->recv_data(inp_in.data(), (col_end - col_start + 1) * sizeof(uint64_t));
        if(verbose_info){
            cout << "[Server] masked input received" << endl;
        }
        //add the share
        for(int i = 0; i < (col_end - col_start + 1); i++){
            inp_in(i,0) = (inp_in(i,0) + outp_final[i + col_start]) % prime_mod;
            inp_in(i,0) = neg_mod((int64_t)inp_in(i,0), prime_mod);
        }
        //perform the dot product
        //the weight values should be small enough to fit uint64_t
        Channel local_xr0k = chan_pt * inp_in;
        //add the bias and noise share
        for(int i = 0; i < out_dim; i++){
            uint64_t bs_temp = 0ULL;
            if(col_start == 0){
                bs_temp = neg_mod((int64_t)bs[i],prime_mod) * pow(2, x_scales);
            }
            uint64_t lo_temp = neg_mod((int64_t)local_xr0k(i,0), prime_mod);
            outFC[i] = (outFC[i] + lo_temp) % prime_mod;
            outFC[i] = (outFC[i] + bs_temp) % prime_mod;
            outFC[i] = neg_mod((int64_t)outFC[i], prime_mod);
        }
        if(verbose_info){
            cout << "[Server] output share formed" << endl;
        }
    }
    long long t_on = time_from(start_online);
    uint64_t oncomm_end = io->counter;
    cout << "fc: Comm. Sent at online (MiB): " << (oncomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"fc: Online Time (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);  
    Comm_solefc += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    Time_solefc += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (oncomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_solefc_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "fc: Comm. Sent & Recv-ed online (MiB): " << (oncomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }
    //verify the output
    #if defined(DEBUG_EXEC)
        if(party == CLIENT){
            //send final share
            io->send_data(outFC, sizeof(uint64_t) * out_dim);
            //form the input image
            Channel imageInp((col_end - col_start + 1), 1);
            for (int i = 0; i < (col_end - col_start + 1); i++) {
                imageInp(i, 0) = neg_mod((int64_t)outp_final[i + col_start], prime_mod);
            }
            //send input data
            io->send_data(imageInp.data(), (col_end - col_start + 1) * sizeof(uint64_t));
        }else{//the server
            //receive final share
            vector<uint64_t> outFC_0(out_dim, 0ULL);
            io->recv_data(outFC_0.data(), sizeof(uint64_t) * out_dim);
            //get the result from final shares
            for(int i = 0; i < out_dim; i++) {
                outFC_0[i] = (outFC_0[i] + outFC[i]) % prime_mod;
                outFC_0[i] = neg_mod((int64_t)outFC_0[i], prime_mod);
            }
            //receive input
            Channel image_in((col_end - col_start + 1),1);
            io->recv_data(image_in.data(), (col_end - col_start + 1) * sizeof(uint64_t));  
            //get the input
            for(int i = 0; i < (col_end - col_start + 1); i++){
                image_in(i,0) = (image_in(i,0) + neg_mod((int64_t)outp_final[i + col_start], prime_mod)) % prime_mod;
                image_in(i,0) = neg_mod((int64_t)image_in(i,0), prime_mod);
            }           
            //get the dot product
            Channel resultFC = chan_pt * image_in;
            //compare the result
            bool pass = true;
            for (int i = 0; i < out_dim; i++) {
                uint64_t bs_temp = 0ULL;
                if(col_start == 0){
                    bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2, x_scales);
                }
                resultFC(i,0) = (neg_mod(resultFC(i,0),(int64_t)prime_mod) + bs_temp) % prime_mod;
                if (outFC_0[i] != neg_mod(resultFC(i,0), (int64_t) prime_mod)){
                    pass = false;
                }
            }
            if (pass) {
                cout << GREEN << "[Server] Successful Online" << RESET << endl;
            }
            else {
                cout << RED << "[Server] Failed Online" << RESET << endl;
                cout << RED << "WARNING: The implementation assumes that the computation performed by the server (on it's model and masked input)" << endl;
                cout << "fits in a 64-bit integer. The failed operation could be a result of overflowing the bound." << RESET << endl;
            }
        }
    #endif
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_, zero_);
    context_ = nullptr;
    encryptor_ = nullptr;
    decryptor_ = nullptr;
    evaluator_ = nullptr;
    encoder_ = nullptr;
    gal_keys_ = nullptr;
    zero_ = nullptr;

    if(party == CLIENT){
        delete[] r0;
        r0 = nullptr;
    }

    io->flush();
    delete io;
    io = nullptr;
    
    return;
}




