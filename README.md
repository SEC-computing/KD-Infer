# KD-Infer

This is the implementation of KD-Infer for efficient private inference that is able to utilize all zeros, duplicates as well as unique values in scaled raw model weights without retraining, and with much more flexibility for input volume at offline.

## Setup and Build

Run the following script to set up and build the project:

```bash
./auto_run_cmake.sh
```

Required packages are listed in `auto_run_cmake.sh`. If automatic installation does not succeed, install the needed packages manually and then rerun the script.

## Network Configuration

Configure the network bandwidth and round trip time using:

```bash
./netconfig.sh [lan/wan]
```

## Model and Input Preparation

Prepare the models and inputs by running the scripts in the `src/ModelAndInput/` directory:

- `inputScale.py`: Generates scaled input data.
- `modelPrepare.py`: Downloads the required models.
- `extractWeights.py`: Extracts and scales the model weights.
- `sampleTest.py`: Obtains predicted labels from raw models.
- `inputBatchsize.py`: Determines the necessary batch size distribution for the models.

## Testing

All test codes are located in the `tests/` folder, and this repo implements the OT in IKNP style which is adopted in CrypTFlow2 while it can be replaced with the VOLE style adopted in Cheetah. Moreover, you can add additional tests by modifying the `CMakeLists.txt` file.

To run the tests, navigate to the `build/bin` directory where the executables are located. Run the following commands in a terminal:

- To simulate the server: `./[test name] r=1`
- To simulate the client: `./[test name] r=2`

Replace `[test name]` with the actual name of the test executable.
