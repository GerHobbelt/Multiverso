#ifndef CONFIGURE_H_
#define CONFIGURE_H_

#include <string>
#include <unordered_map>

struct Configure {
public:
  // parse configs from a config_file
  // \param config_file should contain:
  //  input_size
  //  output_size
  //  other config will use default value
  //  when not provided
  explicit Configure(const std::string& config_file);

  // dimension of each input
  unsigned long long input_size = 0;
  // dimension of each output
  int output_size = 0;

  // is input data sparse
  bool sparse = false;

  // number of epoch when train
  int train_epoch = 1;
  // minibatch size for sgds
  int minibatch_size = 20;
  // buffer size for async reader
  int read_buffer_size = 2048;
  // show statistic time after process each # sample
  // include computation, communication
  int show_time_per_sample = 10000;

  // coefficient for regularization term 
  double regular_coef = 0.0005;
  // initial learning rate
  double learning_rate = 0.8;
  // learning rate = max(1e-3, 
  //      initial - (update count - learning rate coef * minibatch size))
  // one mini-batch update once

  // input data for training
  // use ; to separate different files
  std::string train_file = "train.data";
  // default / weight / ftrl
  // [default]
  //  read text file, each line as:
  //    1. sparse data use libsvm data format
  //      label key:value key:value ...
  //    2. dense data use format as
  //      label value value ...
  // [weight]
  //  the first column is label:weight(double)
  //  others the same to default
  // [bsparse]
  //  for sparse data, read binary file, each sample as:
  //    count(size_t)label(int)weight(double)key(size_t)key(size_t)...
  std::string reader_type = "default";

  // local model when false, 
  // parameter server model when true
  bool use_ps = false;
  // is use pipeline in ps_model
  bool pipeline = true;
  // pull model after each sync_frequency mini-batch
  int sync_frequency = 1;

  bool sync_mode = false;
  int num_records = 0;
  int class_type = 0;
  int K = 2;
  int num_training_workers = 1;
  double learning_rate_coef = 1e6;
  int num_iters = -1;

private:
  enum ValueType {
    kInt = 0, kFloat, kDouble, kString, kBool, kULL,
  };

  void ParseValue(std::unordered_map<std::string, std::string>* mp,
    const std::string& key, void *value, ValueType type);
  void ParseInt(const std::string value, int* vptr);
  void ParseBool(const std::string value, bool* vptr);
  void ParseFloat(const std::string value, float* vptr);
  void ParseDouble(const std::string value, double* vptr);
  void ParseULL(const std::string value, unsigned long long* vptr);
};

#endif  // CONFIGURE_H_
