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

#ifndef BATCH_STREAM_PPM_H
#define BATCH_STREAM_PPM_H
#include <vector>
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include "NvInfer.h"
#include "logger.h"
#include "common.h"

std::string locateFile(const std::string& input);

static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 256;
static constexpr int INPUT_W = 512;
static constexpr int OUTPUT_C = 4;

const char* INPUT_BLOB_NAME = "input_tensor";

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
    {
        mDims = nvinfer1::DimsNCHW{batchSize, 3, 300, 300};
        mImageSize = mDims.c() * mDims.h() * mDims.w();
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.n() * mImageSize, 0);
        mFileLabels.resize(mDims.n(), 0);
        reset(0);
    }

    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.n();
        skip(firstBatch);
    }

    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
            if (mFileBatchPos == mDims.n() && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.n() && mBatchSize % mDims.n() == 0 && mFileBatchPos == mDims.n())
        {
            mFileCount += skipCount * mBatchSize / mDims.n();
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return mBatch.data(); }
    float* getLabels() { return mLabels.data(); }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    nvinfer1::DimsNCHW getDims() const { return mDims; }

private:
    float* getFileBatch() { return mFileBatch.data(); }
    float* getFileLabels() { return mFileLabels.data(); }

    bool update()
    {
        std::vector<std::string> fNames;

        std::ifstream file(locateFile("list.txt"), std::ios::binary);
        if (file)
        {
            gLogInfo << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);
        }
        for (int i = 1; i <= mBatchSize; i++)
        {
            std::string sName;
            std::getline(file, sName);
            sName = sName + ".ppm";

            gLogInfo << "Calibrating with file " << sName << std::endl;
            fNames.emplace_back(sName);
        }
        mFileCount++;

        std::vector<samplesCommon::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(fNames.size());
        for (uint32_t i = 0; i < fNames.size(); ++i)
        {
            readPPMFile(locateFile(fNames[i]), ppms[i]);
        }
        std::vector<float> data(samplesCommon::volume(mDims));

        long int volChl = mDims.h() * mDims.w();

        for (int i = 0, volImg = mDims.c() * mDims.h() * mDims.w(); i < mBatchSize; ++i)
        {
            for (int c = 0; c < mDims.c(); ++c)
            {
                for (int j = 0; j < volChl; ++j)
                {
                    data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * mDims.c() + c]) - 1.0;
                }
            }
        }

        std::copy_n(data.data(), mDims.n() * mImageSize, getFileBatch());

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};

    int mFileCount{0}, mFileBatchPos{0};
    int mImageSize{0};

    nvinfer1::DimsNCHW mDims;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;
};

#endif
