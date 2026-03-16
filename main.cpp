#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

// Preprocessing parameters (match Python)
const float MEAN = 0.5f;
const float STD  = 0.5f;
const int   IMG_SIZE = 32;

// Use OpenCV dnn::blobFromImage for fast, vectorized preprocessing.
// It does: blob = (image - mean) * scalefactor  (NCHW, float32)
inline void preprocess_to_tensor(const cv::Mat& img, ov::Tensor& input_tensor) {
    // (x/255 - 0.5)/0.5  == (x - 127.5)/127.5
    const double scalefactor = 1.0 / 127.5;
    const cv::Scalar mean(127.5, 127.5, 127.5);

    cv::Mat blob;
    cv::dnn::blobFromImage(
        img,
        blob,
        scalefactor,
        cv::Size(IMG_SIZE, IMG_SIZE),
        mean,
        /*swapRB=*/true,   // BGR -> RGB
        /*crop=*/false
    );

    if (!blob.isContinuous())
        blob = blob.clone();

    float* dst = input_tensor.data<float>();
    std::size_t num_elems = blob.total();
    std::memcpy(dst, blob.ptr<float>(), num_elems * sizeof(float));
}

int main() {
    std::string model_path   = "traffic_sign_cnn_212_openvino.xml";
    std::string input_folder = "crops";
    std::string output_root  = "inspection";

    try {
        // -------------------------------
        // Initialize OpenVINO
        // -------------------------------
        ov::Core core;

        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        // If model is dynamic, reshape to [1, 3, 32, 32]
        if (model->input().get_partial_shape().is_dynamic()) {
            model->reshape({1, 3, IMG_SIZE, IMG_SIZE});
        }

        ov::CompiledModel compiled_model = core.compile_model(
            model,
            "CPU",
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::num_streams(1)
        );

        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // Validate input shape
        ov::Tensor input_tensor = infer_request.get_input_tensor();
        ov::Shape input_shape = input_tensor.get_shape(); // [N, C, H, W]

        if (input_shape.size() != 4 || input_shape[0] != 1) {
            std::cerr << "Unexpected input shape: ";
            for (auto d : input_shape) std::cerr << d << " ";
            std::cerr << std::endl;
            return 1;
        }

        const int in_c = static_cast<int>(input_shape[1]);
        const int in_h = static_cast<int>(input_shape[2]);
        const int in_w = static_cast<int>(input_shape[3]);

        if (in_c != 3 || in_h != IMG_SIZE || in_w != IMG_SIZE) {
            std::cerr << "Model input shape is "
                      << in_c << "x" << in_h << "x" << in_w
                      << " but expected 3x32x32.\n";
            return 1;
        }

        // Determine number of classes from output shape
        ov::Output<const ov::Node> output_port = compiled_model.output();
        ov::Shape output_shape = output_port.get_shape(); // e.g. [1, num_classes] or [1, num_classes, 1, 1]
        std::size_t num_classes = 0;

        if (output_shape.size() == 2) {
            // [1, num_classes]
            num_classes = output_shape[1];
        } else if (output_shape.size() == 4) {
            // [1, num_classes, 1, 1]
            num_classes = output_shape[1];
        } else {
            std::cerr << "Unexpected output shape: ";
            for (auto d : output_shape) std::cerr << d << " ";
            std::cerr << std::endl;
            return 1;
        }

        // -------------------------------
        // Collect input images
        // -------------------------------
        std::vector<fs::path> image_files;
        const std::vector<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp"};

        if (!fs::exists(input_folder) || !fs::is_directory(input_folder)) {
            std::cerr << "Input folder does not exist: " << input_folder << std::endl;
            return 1;
        }

        for (const auto& entry : fs::directory_iterator(input_folder)) {
            if (!entry.is_regular_file())
                continue;

            fs::path p = entry.path();
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                image_files.push_back(p);
            }
        }

        if (image_files.empty()) {
            std::cout << "No images found in folder: " << input_folder << std::endl;
            return 0;
        }

        // -------------------------------
        // Timing accumulators
        // -------------------------------
        std::vector<double> preprocess_times;
        std::vector<double> inference_times;
        std::vector<double> postprocess_times;
        std::vector<double> total_times;

        preprocess_times.reserve(image_files.size());
        inference_times.reserve(image_files.size());
        postprocess_times.reserve(image_files.size());
        total_times.reserve(image_files.size());

        // Ensure output root exists
        fs::create_directories(output_root);

        // -------------------------------
        // Main loop over images
        // -------------------------------
        for (const auto& img_path : image_files) {
            cv::Mat img = cv::imread(img_path.string());
            if (img.empty()) {
                std::cerr << "Error: Could not load image " << img_path << ". Skipping.\n";
                continue;
            }

            using clock = std::chrono::high_resolution_clock;

            // ===== Preprocess =====
            auto t0 = clock::now();
            preprocess_to_tensor(img, input_tensor);
            auto t1 = clock::now();
            double t_pre = std::chrono::duration<double, std::milli>(t1 - t0).count();
            preprocess_times.push_back(t_pre);

            // ===== Inference =====
            auto t2 = clock::now();
            infer_request.infer();
            auto t3 = clock::now();
            double t_inf = std::chrono::duration<double, std::milli>(t3 - t2).count();
            inference_times.push_back(t_inf);

            // ===== Postprocess + Save =====
            auto t4 = clock::now();
            ov::Tensor output_tensor = infer_request.get_output_tensor();
            const float* scores = output_tensor.data<const float>();

            // Argmax over classes
            int pred_class = static_cast<int>(
                std::max_element(scores, scores + num_classes) - scores
            );

            fs::path save_dir = fs::path(output_root) / std::to_string(pred_class);
            fs::create_directories(save_dir);

            fs::path save_path = save_dir / img_path.filename();
            cv::imwrite(save_path.string(), img);

            auto t5 = clock::now();
            double t_post = std::chrono::duration<double, std::milli>(t5 - t4).count();
            postprocess_times.push_back(t_post);

            double total = t_pre + t_inf + t_post;
            total_times.push_back(total);

            std::cout << "Image: " << img_path.filename().string() << "\n";
            std::cout << "Predicted class: " << pred_class << "\n";
            std::cout << "Saved to: " << save_path.string() << "\n\n";

            std::cout << "===== TIMING =====\n";
            std::cout << "Preprocess: " << t_pre  << " ms\n";
            std::cout << "Inference:  " << t_inf  << " ms\n";
            std::cout << "Postprocess:" << t_post << " ms\n";
            std::cout << "TOTAL:      " << total << " ms\n\n";
        }

        // -------------------------------
        // Averages (like Python script)
        // -------------------------------
        const std::size_t n = total_times.size();
        if (n == 0) {
            std::cout << "No images were successfully processed.\n";
            return 0;
        }

        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        };

        double avg_pre  = avg(preprocess_times);
        double avg_inf  = avg(inference_times);
        double avg_post = avg(postprocess_times);
        double avg_tot  = avg(total_times);

        std::cout << "Processed " << n << " images.\n";
        std::cout << "Average preprocess time: " << avg_pre  << " ms\n";
        std::cout << "Average inference time:  " << avg_inf  << " ms\n";
        std::cout << "Average postprocess time:" << avg_post << " ms\n";
        std::cout << "Average total time:      " << avg_tot  << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception during OpenVINO inference: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}