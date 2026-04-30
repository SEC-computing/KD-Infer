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

/************* Data Configuration **********/
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
//this function gets the relu with known MSB
void reconv_relu(uint64_t* out_share, uint64_t* inp_share, uint64_t* msb_share, int lnum_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_relu, double &Time_relu, double &Comm_recv_relu);

void sole_relu(uint64_t* relu_z, uint64_t* inp_shrs, int number_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_relu, double &Time_relu, double &Comm_recv_relu);

void sole_div(int div_num, int divisor, int div_const, uint64_t* avrg_pool_l, uint64_t* outp_final_l, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_trunc, double &Time_trunc, double &Comm_recv_trunc);

void first_conv_out(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, string input_addr, string wts_addr, string bs_addr, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_soleconv, double &Time_soleconv, double &Comm_recv_soleconv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void mxp(uint64_t *x, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_mxp, double &Time_mxp, double &Comm_recv_mxp);

void trunc_func(int div_num, int consSF, uint64_t *outp_final, uint64_t *x, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_trunc, double &Time_trunc, double &Comm_recv_trunc);

void reconv_left(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, uint64_t *bool_shr, vector<vector<uint64_t>> &outArr, Image &imageH5, uint64_t *r0, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void reconv_right(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, uint64_t *bool_shr, uint64_t *r0, vector<vector<uint64_t>> &outArr_two, Image &imageH5, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void reconv_normal(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, vector<vector<uint64_t>> &outArr_two, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1);

void final_fc_max(uint64_t *outp_final, int x_scales, int inp_dim, int out_dim, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_solefc, double &Time_solefc, double &Comm_recv_solefc, double &Comm_argmax, double &Time_argmax, double &Comm_argmax_recv);


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
  
    string model_wts_root = "../../src/ModelAndInput/processed_weights_and_biases/";
    string model_inp_root = "../../src/ModelAndInput/";

    /************ Prepare conv-497 ***************/
    vector<vector<uint64_t>> outArr_g;
    first_conv_out(224, 3, 7, 64, 3, 3, 2, model_inp_root+"uint64_inp.bin", model_wts_root+"onnx::Conv_497.bin", model_wts_root+"onnx::Conv_498.bin", generalData_g, outArr_g, offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_soleconv_g, Time_soleconv_g, Comm_recv_soleconv_g);


    /***************maxpooling*******************/
    generalData_g.inp_chans = 64;
    generalData_g.image_h = 112;
    generalData_g.image_w = 112;
    generalData_g.filter_h = 3;
    generalData_g.filter_w = 3;
    generalData_g.stride_h = 2;
    generalData_g.stride_w = 2;
    generalData_g.pad_t = 1;
    generalData_g.pad_b = 1;
    generalData_g.pad_l = 1;
    generalData_g.pad_r = 1;
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
    int mxpool_row = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    uint64_t *x_g = new uint64_t[mxpool_row];
    mxp(x_g, generalData_g, outArr_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_mxp_g, Time_mxp_g, Comm_recv_mxp_g);
    outArr_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
    int div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    int consSF_g = 12;
    uint64_t *outp_final_g = new uint64_t[div_num_g];
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;


    /************ Prepare conv-509***************/
    Image imageh5;
    uint64_t *r0_g;
    int numb_relu_g = 56*56*64;
    uint64_t *bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    reconv_left(56, 64, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_509.bin", model_wts_root+"onnx::Conv_510.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);


    /************ Prepare conv-500***************/
    vector<vector<uint64_t>> outArr_two_g;
    reconv_right(56, 64, 1, 64, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, r0_g, outArr_two_g, imageh5, model_wts_root+"onnx::Conv_500.bin", model_wts_root+"onnx::Conv_501.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    } 
    delete[] bool_shr_g;
    bool_shr_g = nullptr;
    delete[] outp_final_g;
    outp_final_g = nullptr;
    imageh5 = Image();


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();


    /************ Prepare conv-503***************/

    reconv_normal(56, 64, 3, 64, 1, 1, 1, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_503.bin", model_wts_root+"onnx::Conv_504.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();    
    

    /************ Prepare conv-506***************/
    reconv_normal(56, 64, 1, 256, 0, 0, 1, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_506.bin", model_wts_root+"onnx::Conv_507.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;    


    /*****combine reconv 509 and reconv 506******/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 256;    
    for(int i = 0; i < generalData_g.out_chans; i++) {
        for(int j = 0; j < (generalData_g.output_h*generalData_g.output_w); j++) {
            outArr_g[i][j] = (outArr_g[i][j] + outArr_two_g[i][j]) % prime_mod;
        }
    }
    outArr_two_g = vector<vector<uint64_t>>();
    /********************************************/

    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
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


    /************ Prepare conv-512***************/
    numb_relu_g = 56*56*256;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(56, 256, 1, 64, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_512.bin", model_wts_root+"onnx::Conv_513.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();


    /**************get the relu******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 256;
    int number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    uint64_t *relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
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
    

    /************ Prepare conv-515***************/
    reconv_normal(56, 64, 3, 64, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_515.bin", model_wts_root+"onnx::Conv_516.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;    
    

    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
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


    /************ Prepare conv-518***************/
    reconv_normal(56, 64, 1, 256, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_518.bin", model_wts_root+"onnx::Conv_519.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
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


    /****combine relu and reconv-518*************/     
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;
    /********************************************/

    /************ Prepare conv-521***************/
    numb_relu_g = 56*56*256;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }    
    
    reconv_left(56, 256, 1, 64, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_521.bin", model_wts_root+"onnx::Conv_522.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();


    /**************get the relu******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 256;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
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


    /************ Prepare conv-524***************/
    reconv_normal(56, 64, 3, 64, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_524.bin", model_wts_root+"onnx::Conv_525.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 64;
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


    /************ Prepare conv-527***************/
    reconv_normal(56, 64, 1, 256, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_527.bin", model_wts_root+"onnx::Conv_528.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
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


    /*****combine relu and reconv-527************/     
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-539***************/
    numb_relu_g = 56*56*256;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }    
    
    reconv_left(56, 256, 1, 512, 0, 0, 2, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_539.bin", model_wts_root+"onnx::Conv_540.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);


    /************ Prepare conv-530***************/
    reconv_right(56, 256, 1, 128, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, r0_g, outArr_two_g, imageh5, model_wts_root+"onnx::Conv_530.bin", model_wts_root+"onnx::Conv_531.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    } 
    delete[] bool_shr_g;
    bool_shr_g = nullptr;
    delete[] outp_final_g;
    outp_final_g = nullptr;
    imageh5 = Image();


    /***************truncation*******************/
    generalData_g.output_h = 56;
    generalData_g.output_w = 56;
    generalData_g.out_chans = 128;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();


    /************ Prepare conv-533***************/
    reconv_normal(56, 128, 3, 128, 1, 1, 2, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_533.bin", model_wts_root+"onnx::Conv_534.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();   


    /************ Prepare conv-536***************/
    reconv_normal(28, 128, 1, 512, 0, 0, 1, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_536.bin", model_wts_root+"onnx::Conv_537.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /*****combine reconv 539 and reconv 536******/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;    
    for(int i = 0; i < generalData_g.out_chans; i++) {
        for(int j = 0; j < (generalData_g.output_h*generalData_g.output_w); j++) {
            outArr_g[i][j] = (outArr_g[i][j] + outArr_two_g[i][j]) % prime_mod;
        }
    }
    outArr_two_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
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


    /************ Prepare conv-542***************/
    numb_relu_g = 28*28*512;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }

    reconv_left(28, 512, 1, 128, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_542.bin", model_wts_root+"onnx::Conv_543.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();


    /**************get the relu******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-545***************/
    reconv_normal(28, 128, 3, 128, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_545.bin", model_wts_root+"onnx::Conv_546.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-548***************/
    reconv_normal(28, 128, 1, 512, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_548.bin", model_wts_root+"onnx::Conv_549.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
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


    /*****combine relu and reconv-548************/     
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-551***************/
    numb_relu_g = 28*28*512;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }

    reconv_left(28, 512, 1, 128, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_551.bin", model_wts_root+"onnx::Conv_552.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-554***************/
    reconv_normal(28, 128, 3, 128, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_554.bin", model_wts_root+"onnx::Conv_555.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-557***************/
    reconv_normal(28, 128, 1, 512, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_557.bin", model_wts_root+"onnx::Conv_558.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
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


    /****combine relu and reconv-557*************/     
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-560***************/
    numb_relu_g = 28*28*512;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(28, 512, 1, 128, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_560.bin", model_wts_root+"onnx::Conv_561.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-563***************/
    reconv_normal(28, 128, 3, 128, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_563.bin", model_wts_root+"onnx::Conv_564.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr; 


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 128;
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


    /************ Prepare conv-566***************/
    reconv_normal(28, 128, 1, 512, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_566.bin", model_wts_root+"onnx::Conv_567.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
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


    /****combine relu and reconv-566*************/     
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 512;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-578***************/
    numb_relu_g = 28*28*512;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(28, 512, 1, 1024, 0, 0, 2, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_578.bin", model_wts_root+"onnx::Conv_579.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);


    /************ Prepare conv-569***************/
    reconv_right(28, 512, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, r0_g, outArr_two_g, imageh5, model_wts_root+"onnx::Conv_569.bin", model_wts_root+"onnx::Conv_570.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    } 
    delete[] bool_shr_g;
    bool_shr_g = nullptr;
    delete[] outp_final_g;
    outp_final_g = nullptr;
    imageh5 = Image();


    /***************truncation*******************/
    generalData_g.output_h = 28;
    generalData_g.output_w = 28;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();


    /************ Prepare conv-572***************/
    reconv_normal(28, 256, 3, 256, 1, 1, 2, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_572.bin", model_wts_root+"onnx::Conv_573.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 256;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();   


    /************ Prepare conv-575***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_575.bin", model_wts_root+"onnx::Conv_576.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /*****combine reconv 578 and reconv 575******/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;    
    for(int i = 0; i < generalData_g.out_chans; i++) {
        for(int j = 0; j < (generalData_g.output_h*generalData_g.output_w); j++) {
            outArr_g[i][j] = (outArr_g[i][j] + outArr_two_g[i][j]) % prime_mod;
        }
    }
    outArr_two_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /************ Prepare conv-581***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_581.bin", model_wts_root+"onnx::Conv_582.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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
    


    /************ Prepare conv-584***************/
    reconv_normal(14, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_584.bin", model_wts_root+"onnx::Conv_585.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr; 



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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



    /************ Prepare conv-587***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_587.bin", model_wts_root+"onnx::Conv_588.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /****combine relu and reconv-587*************/     
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-590***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_590.bin", model_wts_root+"onnx::Conv_591.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();


    
    /**************get the relu******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-593***************/
    reconv_normal(14, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_593.bin", model_wts_root+"onnx::Conv_594.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-596***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_596.bin", model_wts_root+"onnx::Conv_597.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /*****combine relu and reconv-596************/     
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-599***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_599.bin", model_wts_root+"onnx::Conv_600.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-602***************/
    reconv_normal(14, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_602.bin", model_wts_root+"onnx::Conv_603.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr; 


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-605***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_605.bin", model_wts_root+"onnx::Conv_606.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /*****combine relu and reconv-605************/     
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-608***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_608.bin", model_wts_root+"onnx::Conv_609.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-611***************/
    reconv_normal(14, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_611.bin", model_wts_root+"onnx::Conv_612.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-614***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_614.bin", model_wts_root+"onnx::Conv_615.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /*****combine relu and reconv-614************/     
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;

    /************ Prepare conv-617***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 256, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_617.bin", model_wts_root+"onnx::Conv_618.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;
    


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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

    
    /************ Prepare conv-620***************/
    reconv_normal(14, 256, 3, 256, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_620.bin", model_wts_root+"onnx::Conv_621.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr; 


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
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


    /************ Prepare conv-623***************/
    reconv_normal(14, 256, 1, 1024, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_623.bin", model_wts_root+"onnx::Conv_624.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
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


    /*****combine relu and reconv-623************/     
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 1024;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-635***************/
    numb_relu_g = 14*14*1024;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(14, 1024, 1, 2048, 0, 0, 2, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_635.bin", model_wts_root+"onnx::Conv_636.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);


    /************ Prepare conv-626***************/
    reconv_right(14, 1024, 1, 512, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, r0_g, outArr_two_g, imageh5, model_wts_root+"onnx::Conv_626.bin", model_wts_root+"onnx::Conv_627.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    } 
    delete[] bool_shr_g;
    bool_shr_g = nullptr;
    delete[] outp_final_g;
    outp_final_g = nullptr;
    imageh5 = Image();


    /***************truncation*******************/
    generalData_g.output_h = 14;
    generalData_g.output_w = 14;
    generalData_g.out_chans = 512;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>();


    /************ Prepare conv-629***************/
    reconv_normal(14, 512, 3, 512, 1, 1, 2, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_629.bin", model_wts_root+"onnx::Conv_630.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 512;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    consSF_g = 12;
    outp_final_g = new uint64_t[div_num_g];
    x_g = new uint64_t[div_num_g];
    //reshape the outArr_two
    for(int i = 0; i < div_num_g; i++){
        x_g[i] = outArr_two_g[i / (generalData_g.output_h*generalData_g.output_w)][i % (generalData_g.output_h*generalData_g.output_w)];
    }
    trunc_func(div_num_g, consSF_g, outp_final_g, x_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);
    delete[] x_g;
    x_g = nullptr;    
    outArr_two_g = vector<vector<uint64_t>>(); 


    /************ Prepare conv-632***************/
    reconv_normal(7, 512, 1, 2048, 0, 0, 1, generalData_g, outp_final_g, outArr_two_g, model_wts_root+"onnx::Conv_632.bin", model_wts_root+"onnx::Conv_633.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;  


    /*****combine reconv 635 and reconv 632******/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;    
    for(int i = 0; i < generalData_g.out_chans; i++) {
        for(int j = 0; j < (generalData_g.output_h*generalData_g.output_w); j++) {
            outArr_g[i][j] = (outArr_g[i][j] + outArr_two_g[i][j]) % prime_mod;
        }
    }
    outArr_two_g = vector<vector<uint64_t>>();


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
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


    /************ Prepare conv-638***************/
    numb_relu_g = 7*7*2048;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(7, 2048, 1, 512, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_638.bin", model_wts_root+"onnx::Conv_639.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 512;
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


    /************ Prepare conv-641***************/
    reconv_normal(7, 512, 3, 512, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_641.bin", model_wts_root+"onnx::Conv_642.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr; 


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 512;
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


    /************ Prepare conv-644***************/
    reconv_normal(7, 512, 1, 2048, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_644.bin", model_wts_root+"onnx::Conv_645.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
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
  
    
    /*****combine relu and reconv-644************/     
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /************ Prepare conv-647***************/
    numb_relu_g = 7*7*2048;
    bool_shr_g = new uint64_t[numb_relu_g];//boolean share
    if (party == CLIENT){
        r0_g = new uint64_t[numb_relu_g];//the random share for client
    }
    
    reconv_left(7, 2048, 1, 512, 0, 0, 1, generalData_g, outp_final_g, bool_shr_g, outArr_g, imageh5, r0_g, model_wts_root+"onnx::Conv_647.bin", model_wts_root+"onnx::Conv_648.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);
    if(party == CLIENT){
        delete[] r0_g;
        r0_g = nullptr;
    }
    imageh5 = Image();



    /**************get the relu******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
    number_relu_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[number_relu_g];
    reconv_relu(relu_g, outp_final_g, bool_shr_g, number_relu_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;
    delete[] bool_shr_g;
    bool_shr_g = nullptr;



    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 512;
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


    /************ Prepare conv-650***************/
    reconv_normal(7, 512, 3, 512, 1, 1, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_650.bin", model_wts_root+"onnx::Conv_651.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 512;
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


    /************ Prepare conv-653***************/
    reconv_normal(7, 512, 1, 2048, 0, 0, 1, generalData_g, outp_final_g, outArr_g, model_wts_root+"onnx::Conv_653.bin", model_wts_root+"onnx::Conv_654.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************truncation*******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
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



    /*****combine relu and reconv-653************/     
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    for(int i = 0; i < div_num_g; i++){
        outp_final_g[i] = (outp_final_g[i] + relu_g[i]) % prime_mod;
    }
    delete[] relu_g;
    relu_g = nullptr;


    /**************get the relu******************/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;
    div_num_g = generalData_g.output_h*generalData_g.output_w*generalData_g.out_chans;
    relu_g = new uint64_t[div_num_g];
    sole_relu(relu_g, outp_final_g, div_num_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_relu_g, Time_relu_g, Comm_recv_relu_g);
    delete[] outp_final_g;
    outp_final_g = nullptr;


    /***************sum up each channel**********/
    generalData_g.output_h = 7;
    generalData_g.output_w = 7;
    generalData_g.out_chans = 2048;    
    int avg_pool_num = generalData_g.out_chans;
    uint64_t *avrg_pool = new uint64_t[avg_pool_num];
    for(int i = 0; i < generalData_g.out_chans; i++) {
        int tmp = i * (generalData_g.output_h * generalData_g.output_w);
        avrg_pool[i] = relu_g[tmp];
        for(int j = 1; j < (generalData_g.output_h * generalData_g.output_w); j++) {
            avrg_pool[i] = (avrg_pool[i] + relu_g[tmp+j]) % prime_mod;
        }
    }    
    delete[] relu_g;
    relu_g = nullptr;
    /********************************************/

    /*************div the summed values**********/
    generalData_g.out_chans = 2048;
    div_num_g = 2048;
    int div_bits_g = 6; //for 49, use 6 bits
    int divisor_g = 1ULL<<6; //log2(7*7)=5.61
    outp_final_g = new uint64_t[div_num_g];
    sole_div(div_num_g, divisor_g, div_bits_g, avrg_pool, outp_final_g, onComm_total_g, onTime_total_g, onComm_recv_g, Comm_trunc_g, Time_trunc_g, Comm_recv_trunc_g);

    delete[] avrg_pool;
    avrg_pool = nullptr;


    /****************FC output*******************/
    final_fc_max(outp_final_g, 6, 2048, 1000, model_wts_root+"fc.weight.bin", model_wts_root+"fc.bias.bin", offComm_total_g, onComm_total_g, offTime_total_g, onTime_total_g, offComm_recv_g, onComm_recv_g, Comm_solefc_g, Time_solefc_g, Comm_recv_solefc_g, Comm_argmax_g, Time_argmax_g, Comm_argmax_recv_g);

    delete[] outp_final_g;
    outp_final_g = nullptr;


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
        cout << "          : Comm. Sent & Recv-ed (MiB): " << (offComm_total_g + onComm_total_g + offComm_recv_g + onComm_recv_g- (Comm_relu_g + Comm_recv_relu_g) - (Comm_trunc_g + Comm_recv_trunc_g) - (Comm_argmax_g + Comm_argmax_recv_g) - (Comm_soleconv_g + Comm_recv_soleconv_g) - (Comm_solefc_g + Comm_recv_solefc_g) - (Comm_mxp_g + Comm_recv_mxp_g)) << endl;
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

//this function gets the relu with known MSB
void reconv_relu(uint64_t* out_share, uint64_t* inp_share, uint64_t* msb_share, int lnum_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_relu, double &Time_relu, double &Comm_recv_relu){
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    otpack = new OTPack<NetIO>(io, party, b, l);
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start();    
    ReLUFieldProtocol<NetIO, uint64_t>* relu_oracle;
    relu_oracle = new ReLUFieldProtocol<NetIO, uint64_t>(party, FIELD, io, l, b, prime_mod, otpack);
    relu_oracle->relu_withmsb(out_share, inp_share, msb_share, lnum_relu);
    delete relu_oracle;
    relu_oracle = nullptr;
    long long t_on = time_from(start_online);
    uint64_t msbcomm_end = io->counter;
    cout << "reconv: Comm. Sent for RELU at online (MiB): " << (msbcomm_end - oncomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Online Time for RELU (l=" << l << "; b=" << b << ") " << t_on * 1.0 / 1000 <<" ms"<< endl;
    onComm_total += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    onTime_total += (t_on * 1.0 / 1000);
    Comm_relu += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    Time_relu += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (msbcomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_relu += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed for RELU at online (MiB): " << (msbcomm_end - oncomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }

    //verify the relu output
    #if defined(DEBUG_EXEC)
        switch (party) {
            case sci::ALICE: {
                io->send_data(inp_share, sizeof(uint64_t)*lnum_relu);
                io->send_data(out_share, sizeof(uint64_t)*lnum_relu);
                break;
            }
            case sci::BOB: {
                uint64_t *xi = new uint64_t[lnum_relu];
                uint64_t *zi = new uint64_t[lnum_relu];
                io->recv_data(xi, sizeof(uint64_t)*lnum_relu);
                io->recv_data(zi, sizeof(uint64_t)*lnum_relu);
                for(int i=0; i<lnum_relu; i++){
                    xi[i] = (xi[i] + inp_share[i]) % prime_mod;
                    zi[i] = (zi[i] + out_share[i]) % prime_mod;
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

    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;

    return;
}


void sole_relu(uint64_t* relu_z, uint64_t* inp_shrs, int number_relu, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_relu, double &Time_relu, double &Comm_recv_relu){
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
    Comm_relu += (double(msbcomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    Time_relu += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (msbcomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_relu += (myRecev / (1.0*(1ULL << 20)));
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

void sole_div(int div_num, int divisor, int div_const, uint64_t* avrg_pool_l, uint64_t* outp_final_l, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_trunc, double &Time_trunc, double &Comm_recv_trunc){
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
    Comm_trunc += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    Time_trunc += (div_end * 1.0 / 1000);  
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (divcomm_end - divcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_trunc += (myRecev / (1.0*(1ULL << 20)));
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


void first_conv_out(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, string input_addr, string wts_addr, string bs_addr, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, double &Comm_soleconv, double &Time_soleconv, double &Comm_recv_soleconv, int SC_inp = 12, int SC_wts = 12, int N = 1){
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
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        }
        //rot-free HE computing with plaintext kernel
        vector<Ciphertext> enc_Kr0(CO);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < CO; i++){
            enc_Kr0[i] = *zero_;
            evaluator_->mod_switch_to_next_inplace(enc_Kr0[i]);
            for(int j = 0; j < r0hat_ctNum; j++){
                vector<uint64_t> v_tmp(slot_count, 0ULL);
                Plaintext tmp;
                int chan_offset = j * chanPerCipher;
                if(j == (r0hat_ctNum - 1)){
                    for(int k = 0; k < (col_heightR0 - chan_offset); k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);
                    }
                }else{
                    for(int k = 0; k < chanPerCipher; k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);
                    }
                }
                if(isAllZero(v_tmp)){
                    continue;
                }else{
                    //encode the kernel vector
                    encoder_->encode(v_tmp, tmp);
                    //perform the multiplication
                    Ciphertext tmp_ct;
                    evaluator_->multiply_plain(enc_r0hat[j], tmp, tmp_ct);
                    //add the output
                    evaluator_->add_inplace(enc_Kr0[i], tmp_ct); 
                }           
            }
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
                for(int k = 1; k < chanPerCipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr[i][j] = (outArr[i][j] + shr12off[i][j + idx_offset]) % prime_mod;
                }
                outArr[i][j] = neg_mod((int64_t)outArr[i][j], prime_mod);
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
        int cipherNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> r0hat_ct(cipherNum);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < cipherNum; i++){
            vector<uint64_t> tmp_vec(slot_count, 0ULL);
            Plaintext tmp_pt;
            int chan_offset = i * chanPerCipher;
            if(i == (cipherNum - 1)){
                for(int j = 0; j < (col_heightR0 - chan_offset); j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                }
            }else{
                for(int j = 0; j < chanPerCipher; j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                }
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
                for(int k = 1; k < chanPerCipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr[i][j] = (outArr[i][j] + Kr0result[i][j + idx_offset]) % prime_mod;
                }
            }
        }
        if(verbose_info){
            cout << "[Client] output share formed" << endl;
        }        
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "Conv: Comm. Sent at offline (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"Conv: Offline Time (l=" << l << "; b=" << b << ") " << t_off * 1.0 / 1000 <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / 1000);  
    Comm_soleconv += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    Time_soleconv += (t_off * 1.0 / 1000);  
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_soleconv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "Conv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
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
    Comm_soleconv += (double(oncomm_end - oncomm_start)/(1.0*(1ULL<<20)));
    Time_soleconv += (t_on * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (oncomm_end - oncomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_soleconv += (myRecev / (1.0*(1ULL << 20)));
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

void mxp(uint64_t *x, ConvMetadata &generalData, vector<vector<uint64_t>> &outArr, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_mxp, double &Time_mxp, double &Comm_recv_mxp){
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
    Comm_mxp += (double(mxpcomm_end - mxpcomm_start)/(1.0*(1ULL<<20)));
    Time_mxp += (t_mxp * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (mxpcomm_end - mxpcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));
    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_mxp += (myRecev / (1.0*(1ULL << 20)));
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

void trunc_func(int div_num, int consSF, uint64_t *outp_final, uint64_t *x, double &onComm_total, double &onTime_total, double &onComm_recv, double &Comm_trunc, double &Time_trunc, double &Comm_recv_trunc){
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
    Comm_trunc += (double(divcomm_end - divcomm_start)/(1.0*(1ULL<<20)));
    Time_trunc += (div_end * 1.0 / 1000);
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (divcomm_end - divcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        onComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_recv_trunc += (myRecev / (1.0*(1ULL << 20)));
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

void reconv_left(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, uint64_t *bool_shr, vector<vector<uint64_t>> &outArr, Image &imageH5, uint64_t *r0, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1){
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
    vector<vector<uint64_t>>Kr0result(CO);
    std::vector<uint64_t> wts;
    std::vector<uint64_t> bs; 
    Image imageR0;
    outArr.resize(CO, vector<uint64_t>(newH * newW, 0ULL));
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
    io = new NetIO(party==1 ? nullptr:address.c_str(), port);
    int slot_count = min(SEAL_POLY_MOD_DEGREE_MAX, max(8192, next_pow2(newH*newH)));
    vector<vector<uint64_t>>shr12off(CO, vector<uint64_t>(slot_count, 0ULL));
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

    vector<Ciphertext> enc_g1h3;
    int num_ct_g1h3 = ceil(1.0 * number_relu / slot_count);
    /********************************************/

    /***************reluconv-509 offline*********/
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
        send_encrypted_vector(io, g1h3_ct);        
        if(verbose_info){
            cout << "[Server] encrypted g1 and h3 sent" << endl;
        }
        //reshape the kernel
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
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        } 
        //perform rot-free computation with plaintext kernel
        vector<Ciphertext> enc_Kr0(CO);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < CO; i++){
            enc_Kr0[i] = *zero_;
            evaluator_->mod_switch_to_next_inplace(enc_Kr0[i]);
            for(int j = 0; j < r0hat_ctNum; j++){
                vector<uint64_t> v_tmp(slot_count, 0ULL);
                Plaintext tmp;
                int chan_offset = j * chanPerCipher;
                if(j == (r0hat_ctNum - 1)){
                    for(int k = 0; k < (col_heightR0 - chan_offset); k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);    
                    }
                }else{
                    for(int k = 0; k < chanPerCipher; k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);
                    }
                }                
                if(isAllZero(v_tmp)){
                    continue;
                }else{
                    //encode the kernel vector
                    encoder_->encode(v_tmp, tmp);
                    //perform the multiplication
                    Ciphertext tmp_ct;
                    evaluator_->multiply_plain(enc_r0hat[j], tmp, tmp_ct);
                    //add the output
                    evaluator_->add_inplace(enc_Kr0[i], tmp_ct); 
                }           
            }
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
        //compute shr11off 
        Image imageH4;
        imageH4.resize(CI); 
        #pragma omp parallel for num_threads(numThreads) schedule(static)        
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    if((int64_t)bool_shr[idx] == 1){
                        tmp_chan(h, w) = neg_mod((int64_t)outp_final[idx], prime_mod);
                    }else{
                        tmp_chan(h, w) = 0;
                    }
                }
            }
            imageH4[chan] = tmp_chan;
        }        
        //the filter values should be small enough to fit uint64_t
        Image local_kH4 = ideal_function(imageH4, myFilters_pt, generalData); 
        //reset the shr12off
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < (chanPerCipher * col_widthR0); j++){
                shr12off[i][j] = neg_mod((int64_t)(prime_mod - shr12off[i][j]), prime_mod);
            }
        }
        //gets partial share
        for(int i = 0; i < CO; i++){
            for(int j = 0; j < col_widthR0; j++){
                outArr[i][j] = shr12off[i][j];
                for(int k = 1; k < chanPerCipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr[i][j] = (outArr[i][j] + shr12off[i][j + idx_offset]) % prime_mod;
                }
                outArr[i][j] += neg_mod((int64_t)local_kH4[i](j / generalData.output_w, j % generalData.output_w), prime_mod);
                outArr[i][j] = neg_mod((int64_t)outArr[i][j], prime_mod);
            }
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }else{//the client
        //generate r0
      
        prg.random_mod_p<uint64_t>(r0, number_relu, prime_mod);        
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
        int cipherNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> r0hat_ct(cipherNum);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < cipherNum; i++){
            vector<uint64_t> tmp_vec(slot_count, 0ULL);
            Plaintext tmp_pt;
            int chan_offset = i * chanPerCipher;
            if(i == (cipherNum - 1)){
                for(int j = 0; j < (col_heightR0 - chan_offset); j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                }
            }else{
                for(int j = 0; j < chanPerCipher; j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                    
                }
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
                outArr[i][j] = Kr0result[i][j];
                for(int k = 1; k < generalData.chans_per_cipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr[i][j] = (outArr[i][j] + Kr0result[i][j + idx_offset]) % prime_mod;
                }
                outArr[i][j] = neg_mod((int64_t)outArr[i][j], prime_mod);
            }
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "reconv: Comm. Sent at offline (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Offline Time (l=" << l << "; b=" << b << ") " << t_off * 1.0 / 1000 <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / 1000);    
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }    
    /********************************************/

    /***************reluconv-509 online**********/
    otpack = new OTPack<NetIO>(io, party, b, l);
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
        imageH5.resize(CI);
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)pt_h5[idx / slot_count][idx % slot_count], prime_mod);
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
                outArr[i][j] = (outArr[i][j]+neg_mod((int64_t)local_kH5[i](j / generalData.output_w, j % generalData.output_w), prime_mod)) % prime_mod;
                outArr[i][j] = (outArr[i][j]+bs_temp)% prime_mod;  
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
                    io->send_data(outArr[i].data(), sizeof(uint64_t) * (col_w));
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
                        outArr_0[i][j] = (outArr_0[i][j] + outArr[i][j]) % prime_mod;
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

    delete otpack;
    otpack = nullptr;

    io->flush();
    delete io;
    io = nullptr;
    /********************************************/
    return;
}

void reconv_right(int image_h, int inp_chans, int filter_h, int out_chans, int pad_l, int pad_r, int stride, ConvMetadata &generalData, uint64_t *outp_final, uint64_t *bool_shr, uint64_t *r0, vector<vector<uint64_t>> &outArr_two, Image &imageH5, string wts_addr, string bs_addr, double &offComm_total, double &onComm_total, double &offTime_total, double &onTime_total, double &offComm_recv, double &onComm_recv, int SC_inp = 12, int SC_wts = 12, int N = 1){
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
    Image imageR0;
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
    int num_ct_g1h3 = ceil(1.0 * number_relu / slot_count);
    /********************************************/

    /***************reluconv-500 offline*********/
    uint64_t offcomm_start = io->counter;
    auto start_offline = clock_start();    
    if(party == SERVER){
        //reshape the kernel
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
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        } 
        //perform rot-free computation with plaintext kernel
        vector<Ciphertext> enc_Kr0(CO);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < CO; i++){
            enc_Kr0[i] = *zero_;
            evaluator_->mod_switch_to_next_inplace(enc_Kr0[i]);
            for(int j = 0; j < r0hat_ctNum; j++){
                vector<uint64_t> v_tmp(slot_count, 0ULL);
                Plaintext tmp;
                int chan_offset = j * chanPerCipher;
                if(j == (r0hat_ctNum - 1)){
                    for(int k = 0; k < (col_heightR0 - chan_offset); k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);    
                    }
                }else{
                    for(int k = 0; k < chanPerCipher; k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);
                    }
                }                
                if(isAllZero(v_tmp)){
                    continue;
                }else{
                    //encode the kernel vector
                    encoder_->encode(v_tmp, tmp);
                    //perform the multiplication
                    Ciphertext tmp_ct;
                    evaluator_->multiply_plain(enc_r0hat[j], tmp, tmp_ct);
                    //add the output
                    evaluator_->add_inplace(enc_Kr0[i], tmp_ct); 
                }           
            }
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
        //compute shr11off 
        Image imageH4;
        imageH4.resize(CI); 
        #pragma omp parallel for num_threads(numThreads) schedule(static)        
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    if((int64_t)bool_shr[idx] == 1){
                        tmp_chan(h, w) = neg_mod((int64_t)outp_final[idx], prime_mod);
                    }else{
                        tmp_chan(h, w) = 0;
                    }
                }
            }
            imageH4[chan] = tmp_chan;
        }        
        //the filter values should be small enough to fit uint64_t
        Image local_kH4 = ideal_function(imageH4, myFilters_pt, generalData); 
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
                for(int k = 1; k < chanPerCipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr_two[i][j] = (outArr_two[i][j] + shr12off[i][j + idx_offset]) % prime_mod;
                }
                outArr_two[i][j] = (outArr_two[i][j] + neg_mod((int64_t)local_kH4[i](j / generalData.output_w, j % generalData.output_w), prime_mod)) % prime_mod;
                outArr_two[i][j] = neg_mod((int64_t)outArr_two[i][j], prime_mod);
            }
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }else{//the client
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
        int cipherNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> r0hat_ct(cipherNum);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < cipherNum; i++){
            vector<uint64_t> tmp_vec(slot_count, 0ULL);
            Plaintext tmp_pt;
            int chan_offset = i * chanPerCipher;
            if(i == (cipherNum - 1)){
                for(int j = 0; j < (col_heightR0 - chan_offset); j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                }
            }else{
                for(int j = 0; j < chanPerCipher; j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                    
                }
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
        //get shr0
        for(int i = 0; i < CO; i++) {
            for(int j = 0; j < col_widthR0; j++) {
                outArr_two[i][j] = Kr0result[i][j];
                for(int k = 1; k < generalData.chans_per_cipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr_two[i][j] = (outArr_two[i][j] + Kr0result[i][j + idx_offset]) % prime_mod;
                }
                outArr_two[i][j] = neg_mod((int64_t)outArr_two[i][j], prime_mod);
            }
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "reconv: Comm. Sent at offline (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Offline Time (l=" << l << "; b=" << b << ") " << t_off * 1.0 / 1000 <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / 1000);    
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
    }    
    /********************************************/

    /***************reluconv-500 online**********/
    uint64_t oncomm_start = io->counter;
    auto start_online = clock_start(); 
    if(party == SERVER){
        //do the convolution and the filter values should be small enough to fit uint64_t
        Image local_kH5 = ideal_function(imageH5, myFilters_pt, generalData);
        //form the final share
        const int col_w = generalData.output_h * generalData.output_w;
        for(int i = 0; i < CO; i++){
            uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2,SC_wts);
            for(int j = 0; j < col_w; j++){
                outArr_two[i][j] = (outArr_two[i][j]+neg_mod((int64_t)local_kH5[i](j / generalData.output_w, j % generalData.output_w), prime_mod)) % prime_mod;
                outArr_two[i][j] = (outArr_two[i][j]+bs_temp)% prime_mod;  
            }
        }
        if(verbose_info){cout << "[Server] output share formed" << endl;}
    }
    long long t_on = time_from(start_online);
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

    io->flush();
    delete io;
    io = nullptr;
    /********************************************/
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
        send_encrypted_vector(io, g1h3_ct);        
        if(verbose_info){
            cout << "[Server] encrypted g1 and h3 sent" << endl;
        }
        //reshape the kernel
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
        int r0hat_ctNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> enc_r0hat(r0hat_ctNum);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0 hat received" << endl;
        } 
        //perform rot-free computation with plaintext kernel
        vector<Ciphertext> enc_Kr0(CO);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < CO; i++){
            enc_Kr0[i] = *zero_;
            evaluator_->mod_switch_to_next_inplace(enc_Kr0[i]);
            for(int j = 0; j < r0hat_ctNum; j++){
                vector<uint64_t> v_tmp(slot_count, 0ULL);
                Plaintext tmp;
                int chan_offset = j * chanPerCipher;
                if(j == (r0hat_ctNum - 1)){
                    for(int k = 0; k < (col_heightR0 - chan_offset); k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);    
                    }
                }else{
                    for(int k = 0; k < chanPerCipher; k++){
                        int idx_offset = k * col_widthR0;
                        int idx_CI = (chan_offset + k) / f_size;
                        int idx_FH = ((chan_offset + k) % f_size) / generalData.filter_w;
                        int idx_FW = ((chan_offset + k) % f_size) % generalData.filter_w;
                        vector<uint64_t> v = {0ULL, myFilters_mod[i][idx_CI](idx_FH, idx_FW)};
                        replace(v_tmp.begin() + idx_offset, v_tmp.begin() + idx_offset + col_widthR0, v[0], v[1]);
                    }
                }                
                if(isAllZero(v_tmp)){
                    continue;
                }else{
                    //encode the kernel vector
                    encoder_->encode(v_tmp, tmp);
                    //perform the multiplication
                    Ciphertext tmp_ct;
                    evaluator_->multiply_plain(enc_r0hat[j], tmp, tmp_ct);
                    //add the output
                    evaluator_->add_inplace(enc_Kr0[i], tmp_ct); 
                }           
            }
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
        //compute shr11off 
        Image imageH4;
        imageH4.resize(CI); 
        #pragma omp parallel for num_threads(numThreads) schedule(static)        
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    if((int64_t)bool_shr[idx] == 1){
                        tmp_chan(h, w) = neg_mod((int64_t)outp_final[idx], prime_mod);
                    }else{
                        tmp_chan(h, w) = 0ULL;
                    }
                }
            }
            imageH4[chan] = tmp_chan;
        }        
        //the filter values should be small enough to fit uint64_t
        Image local_kH4 = ideal_function(imageH4, myFilters_pt, generalData); 
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
                for(int k = 1; k < chanPerCipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr_two[i][j] = (outArr_two[i][j] + shr12off[i][j + idx_offset]) % prime_mod;
                }
                outArr_two[i][j] = (outArr_two[i][j]+ neg_mod((int64_t)local_kH4[i](j / generalData.output_w, j % generalData.output_w), prime_mod)) % prime_mod;
                outArr_two[i][j] = neg_mod((int64_t)outArr_two[i][j], prime_mod);
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
        int cipherNum = ceil(1.0 * col_heightR0 / chanPerCipher);
        vector<Ciphertext> r0hat_ct(cipherNum);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < cipherNum; i++){
            vector<uint64_t> tmp_vec(slot_count, 0ULL);
            Plaintext tmp_pt;
            int chan_offset = i * chanPerCipher;
            if(i == (cipherNum - 1)){
                for(int j = 0; j < (col_heightR0 - chan_offset); j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                }
            }else{
                for(int j = 0; j < chanPerCipher; j++){
                    int len_offset = j * col_widthR0;
                    for(int k = 0; k < col_widthR0; k++){
                        tmp_vec[len_offset + k] = image_colR0(chan_offset + j, k);
                    }
                    
                }
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
                for(int k = 1; k < generalData.chans_per_cipher; k++) {
                    int idx_offset = k * col_widthR0;
                    outArr_two[i][j] = (outArr_two[i][j] + Kr0result[i][j + idx_offset]) % prime_mod;
                }
                outArr_two[i][j] = neg_mod((int64_t)outArr_two[i][j], prime_mod);
            }
        }
        if(verbose_info){cout << "[Client] output share formed" << endl;}
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "reconv: Comm. Sent at offline (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"reconv: Offline Time (l=" << l << "; b=" << b << ") " << t_off * 1.0 / 1000 <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / 1000);    
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "reconv: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
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
        for (int chan = 0; chan < CI; chan++) {
            Channel tmp_chan(image_h, W);
            for (int h = 0; h < image_h; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = chan * image_h * W + h * W + w;
                    tmp_chan(h, w) = neg_mod((int64_t)pt_h5[idx / slot_count][idx % slot_count], prime_mod);
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
    vector<uint64_t> noise_fc(slot_count);//for server
    Channel chan_mod(out_dim, inp_dim);
    Channel chan_pt(out_dim, inp_dim);
    if(party == CLIENT){
        r0 = new uint64_t[inp_dim];
        //generate the r0
        prg.random_mod_p<uint64_t>(r0, inp_dim, prime_mod);
        int num_per_cipher = slot_count / out_dim;
        int num_ct_r0 = ceil(1.0*inp_dim / num_per_cipher);
        vector<Ciphertext> enc_r0(num_ct_r0);
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for(int i = 0; i < num_ct_r0; i++){
            vector<uint64_t> v1(slot_count, 0ULL);
            Plaintext tmp;
            int idx_offset = i * num_per_cipher;
            if(i == (num_ct_r0 - 1)){
                for(int j = 0; j < (inp_dim - idx_offset); j++){
                    for(int k = 0; k < out_dim; k++){
                        v1[j * out_dim + k] = neg_mod((int64_t)r0[idx_offset + j],prime_mod);
                    }
                }
            }else{
                for(int j = 0; j < num_per_cipher; j++){
                    for(int k = 0; k < out_dim; k++){
                        v1[j * out_dim + k] = neg_mod((int64_t)r0[idx_offset + j],prime_mod);
                    }
                }
            }
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
        Ciphertext ct_Kr0;
        recv_ciphertext(io, ct_Kr0);
        if(verbose_info){
            cout << "[Client] masked Kr0 received" << endl;
        }
        //decrypt the masked kR0
        Plaintext tmp;
        vector<uint64_t> kr0_out(slot_count, 0ULL);
        decryptor_->decrypt(ct_Kr0, tmp);
        encoder_->decode(tmp, kr0_out);
        if(verbose_info){cout << "[Client] share decrypted" << endl;}
        //get the kr0 share
        for(int i = 0; i < out_dim; i++) {
            outFC[i] = kr0_out[i];
            for (int j = 1; j < num_per_cipher; j++) {
                outFC[i] = (outFC[i] + kr0_out[i + j * out_dim]) % prime_mod;
            }
            outFC[i] = neg_mod((int64_t)outFC[i],prime_mod);
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
        int num_per_cipher = slot_count / out_dim;
        int num_ct_r0 = ceil(1.0*inp_dim / num_per_cipher);
        vector<Ciphertext> enc_r0hat(num_ct_r0);
        recv_encrypted_vector(io, enc_r0hat);
        if(verbose_info){
            cout << "[Server] encrypted r0hat received" << endl;
        }
        //perform rot-free computation
        Ciphertext wts_inp = *zero_;
        evaluator_->mod_switch_to_next_inplace(wts_inp);
        for(int i = 0; i < num_ct_r0; i++){
            vector<uint64_t> v1(slot_count, 0ULL);
            Plaintext tmp;
            int idx_offset = i * num_per_cipher;
            if(i == (num_ct_r0 - 1)){
                for(int j = 0; j < (inp_dim - idx_offset); j++){
                    for(int k = 0; k < out_dim; k++){
                        v1[j * out_dim + k] = chan_mod(k, idx_offset + j);
                    }
                }
            }else{
                for(int j = 0; j < num_per_cipher; j++){
                    for(int k = 0; k < out_dim; k++){
                        v1[j * out_dim + k] = chan_mod(k, idx_offset + j);
                    }
                }
            }
            encoder_->encode(v1, tmp);
            Ciphertext tmp_ct;
            evaluator_->multiply_plain(enc_r0hat[i], tmp, tmp_ct);
            evaluator_->add_inplace(wts_inp, tmp_ct);
        }
        //add the noise
        prg.random_mod_p<uint64_t>(noise_fc.data(), slot_count, prime_mod);
        Plaintext tmp_res;
        encoder_->encode(noise_fc, tmp_res);
        evaluator_->add_plain_inplace(wts_inp, tmp_res);
        //perform the noise flooding
        parms_id_type parms_id = wts_inp.parms_id();
        shared_ptr<const SEALContext::ContextData> context_data
        = context_->get_context_data(parms_id);        
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        flood_ciphertext(wts_inp, context_data, SMUDGING_BITLEN);
        evaluator_->mod_switch_to_next_inplace(wts_inp);
        //send masked kR0hat
        send_ciphertext(io, wts_inp);
        #if defined(DEBUG_EXEC)
            GET_NOISE_BUDGET(decryptor_, wts_inp, "Server", "after mod-switch");
        #endif            
        if(verbose_info){
            cout << "[Server] encrypted share sent" << endl;
        }
        //reset the noise
        for(int i = 0; i < out_dim; i++){
            outFC[i] = neg_mod((int64_t)(prime_mod - noise_fc[i]), prime_mod);
            for(int j = 1; j < num_per_cipher; j++){
                outFC[i] = (outFC[i] + neg_mod((int64_t)(prime_mod - noise_fc[j*out_dim + i]), prime_mod)) % prime_mod;
            }
            outFC[i] = neg_mod((int64_t)outFC[i], prime_mod);
        }
        if(verbose_info){
            cout << "[Server] share generated" << endl;
        }
    }
    long long t_off = time_from(start_offline);
    uint64_t offcomm_end = io->counter;
    cout << "fc: Comm. Sent at offline (MiB): " << (offcomm_end - offcomm_start)/(1.0*(1ULL << 20)) << endl;
    cout <<"fc: Offline Time (l=" << l << "; b=" << b << ") " << t_off * 1.0 / 1000 <<" ms"<< endl;
    offComm_total += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    offTime_total += (t_off * 1.0 / 1000);   
    Comm_solefc += ((offcomm_end - offcomm_start)/(1.0*(1ULL << 20)));
    Time_solefc += (t_off * 1.0 / 1000); 
    if(party == CLIENT){
        //client sends the number of communication
        uint64_t mySent = (offcomm_end - offcomm_start);
        io->send_data(&mySent, sizeof(uint64_t));

    }else{//the server recieves the number
        uint64_t myRecev = 0;
        io->recv_data(&myRecev, sizeof(uint64_t));
        offComm_recv += (myRecev / (1.0*(1ULL << 20)));
        Comm_solefc_recv += (myRecev / (1.0*(1ULL << 20)));
        //cout << "fc: Comm. Sent & Recv-ed at offline (MiB): " << (offcomm_end - offcomm_start + myRecev)/(1.0*(1ULL << 20)) << endl;
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
                uint64_t bs_temp = neg_mod((int64_t)bs[i], prime_mod) * pow(2,6);
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
            cout <<"Comparison Time: "<<GREEN<<(t * 1.0 / 1000)<<" ms" <<RESET<< endl;
            cout << "Argmax communication (MiB): " <<BLUE<< ((double)(comm_end - comm_start))/(1.0*(1ULL << 20))<<RESET<< endl;
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
            auto start_argmax = clock_start();
            argmax_oracle.ArgMaxMPC(out_dim, outFC, argmax_output_protocol_arg, true, argmax_output_protocol);
            long long t_argmax = time_from(start_argmax);
            uint64_t comm_end = io->counter;
            cout << "Argmax communication (MiB): " <<BLUE<< ((double)(comm_end - comm_start))/(1.0*(1ULL << 20)) << RESET<< endl;
            Comm_argmax += ((double)(comm_end - comm_start))/(1.0*(1ULL << 20));
            Time_argmax += (t_argmax * 1.0 / 1000);
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

