#include <iostream>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;
std::vector<size_t> efs = {
    10, 20, 40, 50, 60, 80, 100, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1500
};
size_t test_round = 3;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: degree bound for symqg, must be a multiple of 32\n"
                  << "arg3: ef for indexing \n"
                  << "arg4: path for saving index\n"
                  << "arg5: metric type (\"l2\" or \"ip\"), l2 by default\n";
        exit(1);
    }

    char* data_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];
    size_t degree = atoi(argv[4]);
    size_t ef = atoi(argv[5]);

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    // if (argc > 5) {
    //     std::string metric_str(argv[5]);
    //     if (metric_str == "ip" || metric_str == "IP") {
    //         metric_type = rabitqlib::METRIC_IP;
    //     }
    // }
    // if (metric_type == rabitqlib::METRIC_IP) {
    //     std::cout << "Metric Type: IP\n";
    // } else if (metric_type == rabitqlib::METRIC_L2) {
    //     std::cout << "Metric Type: L2\n";
    // }

    data_type data;
    data_type query;
    gt_type gt;

    rabitqlib::load_vecs<float, data_type>(data_file, data);
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    int topk = gt.cols();
    int nq = query.rows();
    size_t total_count = nq * topk;
    rabitqlib::StopW stopw;

    index_type qg(data.rows(), data.cols(), degree, metric_type);

    rabitqlib::symqg::QGBuilder builder(qg, ef, data.data());

    // 3 iters, refine at last iter
    builder.build();

    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

    for (size_t r = 0; r < test_round; r++) {
        for (size_t i = 0; i < efs.size(); ++i) {
            size_t ef = efs[i];
            size_t total_correct = 0;
            float total_time = 0;
            qg.set_ef(ef);
            std::vector<PID> results(topk);
            for (size_t z = 0; z < nq; z++) {
                stopw.reset();
                qg.search(&query(z, 0), topk, results.data());
                total_time += stopw.get_elapsed_micro();
                for (size_t y = 0; y < topk; y++) {
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(z, k) == results[y]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "EF\tQPS\tRecall\n";
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
    }


    

    return 0;
}
