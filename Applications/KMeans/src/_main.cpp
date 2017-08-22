#include "multiverso/multiverso.h"
#include "multiverso/table/array_table.h"
#include "reader.h"
#include "configure.h"

#include <string>
#include <iostream>
#include <chrono>
#include <sstream>

class KMeans {
 public:
  KMeans(const std::string& config_file): train_data_(nullptr) {
    config_ = new Configure(config_file);
  }

  void Load() {
    // Get data partition
    int total_size = config_->num_records;
    int num_workers = multiverso::MV_NumWorkers();
    int partition_size = total_size / num_workers;
    int id = multiverso::MV_WorkerId();
    if (id < total_size % num_workers) {
        train_data_ = new DataStore<float>(config_->train_file,
                config_->input_size, config_->output_size,
                id * (partition_size + 1), partition_size + 1,
                config_->class_type);
    } else {
        train_data_ = new DataStore<float>(config_->train_file,
                config_->input_size, config_->output_size,
                id * partition_size + total_size % num_workers, partition_size,
                config_->class_type);
    }
    train_data_->Load();

    multiverso::ArrayTableOption<float> option(config_->input_size);
    table_ = multiverso::MV_CreateTable(option);
  }
  void Train() {
    {
      // Add/Get example
      std::vector<float> delta(config_->input_size, 1);
      table_->Add(delta.data(), delta.size());

      std::vector<float> model(config_->input_size, 0);
      table_->Get(model.data(), model.size());
      Log::Write(Info, "model: %f\n", model[0]);
    }

    Log::Write(Info, "training");
    int batch_size = 10;
    Sample<float>** samples = new Sample<float>*[batch_size];
    SparseBlock<bool> keys;
    for (int iter = 0; iter < 1; ++ iter) {
      train_data_->Read(batch_size, samples, &keys);
      for (int i = 0; i < batch_size; ++ i) {
        Sample<float>* sample = samples[i];
        std::ostringstream os;
        for (int j = 0; j < sample->keys.size(); ++ j) {
          // std::cout << "key: " << sample->keys[i] << " " << sample->values[i];
          os << sample->keys[j] << ":" << sample->values[j] << " ";
        }
        std::cout << os.str() << std::endl;
      }
    }
  }
 private:
  DataStore<float>* train_data_;
  Configure* config_;
  multiverso::ArrayWorker<float>* table_;
};

int main(int argc, char* argv[]) {
  multiverso::MV_SetFlag("sync", true);
  multiverso::MV_Init();
  KMeans kmeans(argv[1]);

  auto start = std::chrono::steady_clock::now();
  kmeans.Load();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  Log::Write(Info, "\033[1;32m[Worker %d] Loading time: %dms.\033[0m\n", multiverso::MV_WorkerId(), duration);

  start = std::chrono::steady_clock::now();
  kmeans.Train();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  Log::Write(Info, "\033[1;32m[Worker %d] Training time: %dms.\033[0m\n", multiverso::MV_WorkerId(), duration);

  multiverso::MV_ShutDown();
  return 0;
}
