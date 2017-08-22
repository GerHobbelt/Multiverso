#ifndef READER_H_
#define READER_H_

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "data_type.h"
#include "util/timer.h"
#include "multiverso/io/io.h"
#include "multiverso/util/mt_queue.h"

// Stores all data read from file
template<typename EleType>
class DataStore {
   public:
    DataStore(
        const std::string& file,
        size_t row_size,  // number of features
        int output_size,  // number of classes
        int line_offset,  // read from a specified sample
        int num_lines,    // size of sample partition to read
        int type = 0,
        bool sparse = true) :
    file_(file),
    row_size_(row_size),
    line_offset_(line_offset),
    output_size_(output_size),
    num_row_(num_lines),
    sparse_(sparse),
    type_(type),
    current_read_(0) {
        buffer_.reserve(num_lines);
    }

    void Read(int num_row, Sample<EleType>** buffer, SparseBlock<bool> *keys);
    void ReadAll(std::vector<Sample<EleType>>** buffer_ptr, SparseBlock<bool>** keys_ptr);
    int Load();

   protected:
    Sample<EleType>* ParseLine(const std::string& line);

    using TextReader = multiverso::TextReader;
    using URI = multiverso::URI;

    Timer timer_;

    int line_offset_;
    std::string file_;

    int num_row_;
    size_t row_size_;
    int output_size_;
    bool sparse_;
    std::vector<Sample<EleType>> buffer_;
    SparseBlock<bool> keys_;
    int type_;  // 1 for linear regression, 0 set negative label as 0, -1 set negative label as -1

    int current_read_;
};

// an async reader for reading matrix data
// each line as a row
template<typename EleType>
class SampleReader {
 public:
  // \param file each line as a row
  // \param row_size num of elements per line
  // \param max_row_buffer_count
  SampleReader(
    const std::string&files,
    size_t row_size,
    int output_size,
    size_t update_per_sample,  // = minibatch_size * sync_frequency
    int max_row_buffer_count,
    bool sparse,
    bool init = true);
  virtual ~SampleReader();
  // \param num_row load at least num_row lines
  // \param buffer put line pointer in buffer
  //  don't free line vector data
  // \return num of rows actually load
  int Read(int num_row, Sample<EleType>**buffer);
  // free read data when won't use
  void Free(int num_row);
  // \return current unread buffer size
  int Ask();
  virtual void Reset();
  inline bool EndOfFile() const { return eof_; }
  multiverso::MtQueue<SparseBlock<bool>*>* keys() { return &keys_; }
  // factory method
  static SampleReader<EleType>* Get(
    const std::string&type,
    const std::string&files,
    size_t row_size,
    int output_size,
    size_t update_per_sample,
    int max_row_buffer_count,
    bool sparse);

protected:
  virtual void Main();
  virtual void ParseLine(const std::string&line, int idx);
  void DeleteKeys();

protected:
  using TextReader = multiverso::TextReader;
  using URI = multiverso::URI;

  Sample<EleType>** buffer_;
  bool sparse_;

  bool eof_;
  TextReader *reader_;
  int reading_file_;
  std::vector<std::string> files_;

  std::thread* thread_;
  int start_;
  int length_;
  int end_;
  int read_length_;

  size_t row_size_;
  int output_size_;
  int buffer_size_;

  multiverso::MtQueue<SparseBlock<bool>*> keys_;
  SparseBlock<bool> *cur_keys_;
  size_t sample_batch_size_;
  size_t sample_count_;

  std::mutex mutex_;
  std::condition_variable cv_;

  Timer timer_;
};  // class SampleReader

template<typename EleType>
class WeightedSampleReader : public SampleReader<EleType> {
 public:
  WeightedSampleReader(
  const std::string&files,
  size_t row_size,
  int output_size,
  size_t update_per_sample,  // = minibatch_size * sync_frequency
  int max_row_buffer_count,
  bool sparse) :
  SampleReader<EleType>(files, row_size, output_size,
    update_per_sample, max_row_buffer_count, sparse) {}

private:
  using TextReader = multiverso::TextReader;
  using URI = multiverso::URI;

private:
  void ParseLine(const std::string&line, int idx);
};  // class WeightedSampleReader

// a binary reader for sparse sample
template<typename EleType>
class BSparseSampleReader : public SampleReader<EleType> {
 public:
   BSparseSampleReader(
    const std::string&files,
    size_t row_size,
    int output_size,
    size_t update_per_sample,
    int max_row_buffer_count,
    bool sparse);
   ~BSparseSampleReader();
  void Reset();

private:
  void Main();
  bool ParseSample(int idx);
  int LoadDataChunk();

private:
  using StreamFactory = multiverso::StreamFactory;
  using FileOpenMode = multiverso::FileOpenMode;
  using TextReader = multiverso::TextReader;
  using URI = multiverso::URI;
  multiverso::Stream *stream_;

  int chunk_idx_;
  int chunk_size_;
  std::vector<char> data_chunk_;
  const int chunk_capacity_ = 1 << 20;
};  // class BSparseSampleReader

#endif  // READER_H_
