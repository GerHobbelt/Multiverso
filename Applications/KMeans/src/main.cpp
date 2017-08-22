#include "multiverso/multiverso.h"
#include "multiverso/table/array_table.h"
#include "reader.h"
#include "configure.h"

#include <string>
#include <iostream>
#include <chrono>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>


// return ID of cluster whose center is the nearest (uses euclidean distance), and the distance
std::pair<int, float> get_nearest_center(const Sample<float>* sample, int K,
                                         const std::vector<float>& params, int num_features) {
    float square_dist, min_square_dist = std::numeric_limits<float>::max();
    int id_cluster_center = -1;

    for (int i = 0; i < K; i++)  // calculate the dist between point and clusters[i]
    {
        typename std::vector<float>::const_iterator first = params.begin() + i * num_features;
        typename std::vector<float>::const_iterator last = first + num_features;
        std::vector<float> diff(first, last);

        for (unsigned int j = 0; j < sample->keys.size(); j++)
            diff[sample->keys[j]] -= sample->values[j];

        square_dist = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

        if (square_dist < min_square_dist) {
            min_square_dist = square_dist;
            id_cluster_center = i;
        }
    }

    return std::make_pair(id_cluster_center, min_square_dist);
}


// test the Sum of Square Error of the model
void test_error(const std::vector<float>& params, Sample<float>* samples[],
                int iter, int K, int data_size, int num_features) {
    
    float sum = 0;  // sum of square error
    std::pair<int, float> id_dist;
    std::vector<int> count(2);

    for (int i = 0; i < data_size; i++) {
        // get next data
        Sample<float>* sample = samples[i];
        id_dist = get_nearest_center(sample, K, params, num_features);
        sum += id_dist.second;
        count[id_dist.first]++;
    }

    std::cout << "Iter " + std::to_string(iter) << ":Within Set Sum of Squared Errors = " << std::to_string(sum) << std::endl;
    for (int i = 0; i < K; i++)  // for tuning learning rate
        std::cout << "count" + std::to_string(i) + ": " + std::to_string(count[i]) << std::endl;
}


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

    multiverso::ArrayTableOption<float> option(config_->input_size * config_->K + config_->K);
    table_ = multiverso::MV_CreateTable(option);
  }
  void Init() {
    {
      if (multiverso::MV_WorkerId() == 0){
        // Random Init
        std::vector<float> params(config_->input_size * config_->K + config_->K, 0);

        int sample_size = config_->input_size;
        Sample<float>** samples = new Sample<float>*[sample_size];
        SparseBlock<bool> keys;
        train_data_->Read(sample_size, samples, &keys);

        //Log::Write(Info, "K: %d\n", config_->K);

        std::vector<int> prohibited_indexes;
        int index;
        for (int i = 0; i < config_->K; i++) {
            while (true) {
                srand(time(NULL));
                index = rand() % sample_size;
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index) ==
                    prohibited_indexes.end())  // not found, this index can be used
                {
                    prohibited_indexes.push_back(index);
                    Sample<float>* sample = samples[index];
                    for (unsigned int j = 0; j < sample->keys.size(); ++ j){
                      params[sample->keys[j] + i * config_->input_size] = sample->values[j];
                    }
                    break;
                }
            }
            params[config_->K * config_->input_size + i] += 1;
        }

        table_->Add(params.data(), params.size());


        // check config
        Log::Write(Info, "config_->K: %d\n", config_->K);
        Log::Write(Info, "config_->num_iters: %d\n", config_->num_iters);
        Log::Write(Info, "config_->num_training_workers: %d\n", config_->num_training_workers);
        Log::Write(Info, "config_->minibatch_size: %d\n", config_->minibatch_size);
        Log::Write(Info, "config_->learning_rate_coef: %d\n", config_->learning_rate_coef);
      }
      else{
        // use only one worker to init and other workers do nothing
        std::vector<float> params(config_->input_size * config_->K + config_->K, 0);
        table_->Add(params.data(), params.size());
      }
    }

  }
  void Train() {
    /*{
      // Check the initial params

      std::vector<float> model(config_->input_size * config_->K + config_->K, 0);
      table_->Get(model.data(), model.size());
      std::ostringstream os;
      for (unsigned int j = 0; j < model.size(); ++ j)
        os << model[j] << " ";

      if (multiverso::MV_WorkerId() == 0)
        std::cout << os.str() << std::endl;
    }*/


    // training task
    int id_nearest_center;
    float alpha;
    //int batch_size = config_->minibatch_size / config_->num_training_workers;
    std::vector<float> model(config_->input_size * config_->K + config_->K, 0);
    std::vector<int> batch_sizes = {100, 100, 200};

    // for test
    Sample<float>** all_samples = new Sample<float>*[config_->num_records];
    SparseBlock<bool> all_keys;

    for (int stage = 0; stage < 3; stage++){
      if (multiverso::MV_WorkerId() == 0)
        std::cout << "Stage " << stage << std::endl;

      int batch_size = batch_sizes[stage] / config_->num_training_workers;
      Sample<float>** samples = new Sample<float>*[batch_size];
      SparseBlock<bool> keys;

      auto start = std::chrono::steady_clock::now();

      for (int iter = 0; iter < config_->num_iters; iter++) {
        table_->Get(model.data(), model.size());
        std::vector<float> step_sum(model);

        // training A mini-batch
        train_data_->Read(batch_size, samples, &keys);

        for (int i = 0; i < batch_size; ++i) {

            Sample<float>* sample = samples[i];
            id_nearest_center = get_nearest_center(sample, config_->K, step_sum, config_->input_size).first;
            alpha = config_->learning_rate_coef / ++step_sum[config_->K * config_->input_size + id_nearest_center];

            std::vector<float>::const_iterator first = step_sum.begin() + id_nearest_center * config_->input_size;
            std::vector<float>::const_iterator last = step_sum.begin() + (id_nearest_center + 1) * config_->input_size;
            std::vector<float> c(first, last);

            for (unsigned int j = 0; j < sample->keys.size(); ++ j)
              c[sample->keys[j]] -= sample->values[j];

            for (unsigned int k = 0; k < config_->input_size; k++)
                step_sum[k + id_nearest_center * config_->input_size] -= alpha * c[k];
        }

        // update model
        for (unsigned int i = 0; i < step_sum.size(); i++)
            step_sum[i] -= model[i];

        table_->Add(step_sum.data(), step_sum.size());



        // test model
        if (iter == config_->num_iters - 1 && multiverso::MV_WorkerId() == 0) {
            train_data_->Read(config_->num_records, all_samples, &all_keys);
            test_error(model, all_samples, iter, config_->K, config_->num_records, config_->input_size);
        }
      }

      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      if (multiverso::MV_WorkerId() == 0)
        Log::Write(Info, "Stage %d training time: %dms.\n", stage, duration);
    }


    /*{
      // Check the final model
      std::vector<float> model(config_->input_size * config_->K + config_->K, 0);
      table_->Get(model.data(), model.size());
      std::ostringstream os;
      for (unsigned int j = 0; j < model.size(); ++ j)
        os << model[j] << " ";

      if (multiverso::MV_WorkerId() == 0)
        std::cout << os.str() << std::endl;
    }*/

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
  kmeans.Init();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  Log::Write(Info, "\033[1;32m[Worker %d] Init time: %dms.\033[0m\n", multiverso::MV_WorkerId(), duration);

  start = std::chrono::steady_clock::now();
  kmeans.Train();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  Log::Write(Info, "\033[1;32m[Worker %d] Training time: %dms.\033[0m\n", multiverso::MV_WorkerId(), duration);

  multiverso::MV_ShutDown();
  return 0;
}
