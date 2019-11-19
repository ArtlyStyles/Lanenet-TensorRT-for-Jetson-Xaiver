/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "BatchStreamPPM.h"
#include "NvUffParser.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "NvInferPlugin.h"
#include "EntropyCalibrator.h"

using namespace nvinfer1;
using namespace nvuffparser;

const std::string gSampleName = "TensorRT.sample_uff_lanenet";

static samplesCommon::Args gArgs;

static constexpr int OUTPUT_CLS_SIZE = 91;

const char* OUTPUT_BLOB_NAME01 = "lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/Conv2D";
const char* OUTPUT_BLOB_NAME0 = "lanenet_model/vgg_backend/binary_seg/ArgMax";

//INT8 Calibration, currently set to calibrate over 100 images
static constexpr int CAL_BATCH_SIZE = 10;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 100, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};

// Visualization
const float visualizeThreshold = 0.5;

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    assert(samplesCommon::getElementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * samplesCommon::getElementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpyAsync(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs, std::max_element(outputs, outputs + eltCount));

    gLogVerbose << "Output:\n";
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        gLogVerbose << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            gLogVerbose << "***";
        gLogVerbose << "\n";
    }
    gLogVerbose << std::endl;

    delete[] outputs;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/lanenet/",
                                  "data/ssd/VOC2007/",
                                  "data/ssd/VOC2007/PPMImages/",
                                  "data/samples/sampleUffLanenet/",
                                  "data/int8_samples/ssd/",
                                  "int8/ssd/",
                                  "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

void populateTFInputData(float* data)
{

    auto fileName = locateFile("inp_bus.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        istringstream iss(line);
        float num;
        iss >> num;
        data[id++] = num;
    }

    return;
}

void populateClassLabels(std::string (&CLASSES)[OUTPUT_CLS_SIZE])
{

    auto fileName = locateFile("ssd_coco_labels.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        CLASSES[id++] = line;
    }

    return;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, IInt8Calibrator* calibrator, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();

    gLogInfo << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }

    gLogInfo << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    if (gArgs.runInInt8)
    {
        builder->setInt8Mode(gArgs.runInInt8);
        builder->setInt8Calibrator(calibrator);
    }
        
    builder->setFp16Mode(gArgs.runInFp16);
    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    gLogInfo << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return nullptr;
    }
    gLogInfo << "End building engine..." << std::endl;

    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();

    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
}

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{

    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();
    gLogInfo <<"nbBindings:" << nbBindings << std::endl;

    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
	gLogInfo << "buffersSize[" << i << "]=" << bufferSizesOutput.first << "*" << sizeof(bufferSizesOutput.second) << std::endl;
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = outputIndex0 + 1;//engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    gLogInfo << "inputIndex:" << inputIndex  << "outputIndex0:" << outputIndex0 << std::endl;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    gLogInfo << "Before exe" << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    context.execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    gLogInfo << "Time taken for inference is " << total << " ms." << std::endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                    buffers[bindingIdx]);
    }

    //CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * OUTPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gLogInfo << "Before of sync" << std::endl;
    cudaStreamSynchronize(stream);
    gLogInfo << "Out of sync" << std::endl;
    
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    //CHECK(cudaFree(buffers[outputIndex1]));
    
    gLogInfo << "Out of free" << std::endl;
}

class FlattenConcat : public IPluginV2
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
    {
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    }
    //clone constructor
    FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, int* inputConcatAxis)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
        , mOutputConcatAxis(outputConcatAxis)
        , mNumInputs(numInputs)
    {
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        for (int i = 0; i < mNumInputs; ++i)
            mInputConcatAxis[i] = inputConcatAxis[i];
    }

    FlattenConcat(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        mIgnoreBatch = read<bool>(d);
        mConcatAxisID = read<int>(d);
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
        mOutputConcatAxis = read<int>(d);
        mNumInputs = read<int>(d);
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

        std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

        mCHW = read<nvinfer1::DimsCHW>(d);

        std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

        assert(d == a + length);
    }
    ~FlattenConcat()
    {
        if (mInputConcatAxis)
            CHECK(cudaFreeHost(mInputConcatAxis));
        if (mCopySize)
            CHECK(cudaFreeHost(mCopySize));
    }
    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims >= 1);
        assert(index == 0);
        mNumInputs = nbInputDims;
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        mOutputConcatAxis = 0;

        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            assert(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1)
                assert(inputs[i].d[0] == inputs[0].d[0]);
            if (mConcatAxisID != 2)
                assert(inputs[i].d[1] == inputs[0].d[1]);
            if (mConcatAxisID != 3)
                assert(inputs[i].d[2] == inputs[0].d[2]);
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 3 ? mOutputConcatAxis : 1);
    }

    int initialize() override
    {
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int numConcats = 1;
        assert(mConcatAxisID != 0);
        numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());
        cublasSetStream(mCublas, stream);

        if (!mIgnoreBatch)
            numConcats *= batchSize;

        float* output = reinterpret_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            float* inputTemp;
            CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

            CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

            for (int n = 0; n < numConcats; ++n)
            {
                CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
            }
            CHECK(cudaFree(inputTemp));
            offset += mInputConcatAxis[i];
        }

        return 0;
    }

    size_t getSerializationSize() const override
    {
        return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
    }

    void serialize(void* buffer) const override
    {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mIgnoreBatch);
        write(d, mConcatAxisID);
        write(d, mOutputConcatAxis);
        write(d, mNumInputs);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mInputConcatAxis[i]);
        }
        write(d, mCHW);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mCopySize[i]);
        }
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        assert(nbOutputs == 1);
        mCHW = inputs[0];
        assert(inputs[0].nbDims == 3);
        CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
        }
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    }
    const char* getPluginType() const override { return "FlattenConcat_TRT"; }

    const char* getPluginVersion() const override { return "1"; }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize = nullptr;
    bool mIgnoreBatch{false};
    int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
    int* mInputConcatAxis = nullptr;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
    std::string mNamespace;
};

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
} // namespace

class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "axis"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mConcatAxisID = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "ignoreBatch"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
            }
        }

        return new FlattenConcat(mConcatAxisID, mIgnoreBatch);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {

        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{false};
    int mConcatAxisID;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);

void printHelp(const char* name)
{
    std::cout << "Usage: " << name << "\n"
        << "Optional Parameters:\n"
        << "  -h, --help Display help information.\n"
        << "  --useDLACore=N    Specify the DLA engine to run on.\n"
        << "  --fp16            Specify to run in fp16 mode.\n"
        << "  --int8            Specify to run in int8 mode." << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments.
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);

    if (gArgs.help)
    {
        printHelp(argv[0]);
        return EXIT_SUCCESS;        
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelp(argv[0]);
        return EXIT_FAILURE;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    auto fileName = locateFile("lanenet_segment_model.uff");
    gLogInfo << fileName << std::endl;

    const int N = 1;
    auto parser = createUffParser();

    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

    parser->registerInput("input_tensor", DimsCHW(3, 256, 512), UffInputOrder::kNCHW);
    // MarkOutput_0 is a node created by the UFF converter when we specify an ouput with -O.
    //parser->registerOutput("lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/Conv2D");
    parser->registerOutput("lanenet_model/vgg_backend/binary_seg/ArgMax");
    
    IHostMemory* trtModelStream{nullptr};

    std::unique_ptr<IInt8Calibrator> calibrator;
    calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, FIRST_CAL_BATCH, "UffSSD", INPUT_BLOB_NAME));

    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, calibrator.get(), trtModelStream);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();

    // Available images.
    std::vector<std::string> imageList = {"0_512_256.ppm"};
    std::vector<samplesCommon::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(N);

    assert(ppms.size() <= imageList.size());
    gLogInfo << " Num batches  " << N << std::endl;
    for (int i = 0; i < N; ++i)
    {
        readPPMFile(locateFile(imageList[i]), ppms[i]);
    }

    vector<float> data(N * INPUT_C * INPUT_H * INPUT_W);

    for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
    {
        for (int c = 0; c < INPUT_C; ++c)
        {
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
            {
	        data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * INPUT_C + INPUT_C - 1 - c]) - 1.0;
            }
        }
    }
    gLogInfo << " Data Size  " << data.size() << std::endl;

    // Deserialize the engine.
    gLogInfo << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    // Host memory for outputs.
    vector<float> detectionOut(N * INPUT_H * INPUT_W);
    vector<int> keepCount(N);

    // Run inference.
    gLogInfo << " Start inference " << keepCount[0] << std::endl;
    doInference(*context, &data[0], &detectionOut[0], &keepCount[0], N);
    gLogInfo << " KeepCount " << keepCount[0] << std::endl;

    bool pass = true;

    for (int i = 0, volImg = INPUT_H * INPUT_W; i < N; ++i)
    {
        for (int c = 0; c < 1; ++c)
        {
            float dmin = 100000, dmax = -100000;
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
            {
	        if (j > 200 * INPUT_W && j < 200 * INPUT_W + 10){
		    gLogInfo << detectionOut[i * volImg + j * OUTPUT_C + c] << " ";
	        }
		
		dmin = std::min(dmin, detectionOut[i * volImg + j * OUTPUT_C + c]);
		dmax = std::max(dmax, detectionOut[i * volImg + j * OUTPUT_C + c]);
	    }
	    gLogInfo << std::endl;
	    
    	    gLogInfo << "min:" << dmin << " max:" << dmax << std::endl;


            float range = dmax - dmin;
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
            {
	         ppms[i].buffer[j * INPUT_C + 0] = (unsigned char)((detectionOut[i * volImg + j * OUTPUT_C + c] - dmin) * 255 / range + 0.5);
		 ppms[i].buffer[j * INPUT_C + 1] = ppms[i].buffer[j * INPUT_C + 0];
		 ppms[i].buffer[j * INPUT_C + 2] = ppms[i].buffer[j * INPUT_C + 0];
            }
        }
    
	writePPMFile("out.ppm", ppms[i]);
    }

    // Destroy the engine.
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return gLogger.reportTest(sampleTest, pass);
}
