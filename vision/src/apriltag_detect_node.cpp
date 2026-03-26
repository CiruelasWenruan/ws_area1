#include <filesystem>
#include <string>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "apriltag.h"

// 定义彩色图像订阅节点类，继承自 rclcpp::Node
class apriltagDetectNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数：初始化节点、订阅者、CV桥接器
     */
    ColorImageSubscriber() : Node("apriltagDetectNode"), image_count_(0)
    {
		//在launch文件里设置导入参数
		this->declare_parameters<bool>("get_pose_en", true);
		this->declare_parameter<std::pmr::string>("apriltagfamily", "tag36h11");

		this->declare_parameter<double>("camera_intrinsic.fx", 615.0);  // 水平焦距
        this->declare_parameter<double>("camera_intrinsic.fy", 615.0);  // 垂直焦距
        this->declare_parameter<double>("camera_intrinsic.cx", 320.0);  // 主点x坐标（图像中心）
        this->declare_parameter<double>("camera_intrinsic.cy", 240.0);  // 主点y坐标（图像中心）

        // 畸变系数（径向k1/k2/k3 + 切向p1/p2，Realsense畸变很小，默认0）
        this->declare_parameter<double>("camera_intrinsic.k1", 0.0);
        this->declare_parameter<double>("camera_intrinsic.k2", 0.0);
        this->declare_parameter<double>("camera_intrinsic.p1", 0.0);
        this->declare_parameter<double>("camera_intrinsic.p2", 0.0);
        this->declare_parameter<double>("camera_intrinsic.k3", 0.0);

        //apriltag检测器参数
        this->declare_parameter<std::string>("family_name","tag36h11" );
        this->declare_parameter<double>("decimate",0.0);
        this->declare_parameter<double>("blur",0.0);
        this->declare_parameter<int>("refine_edges",0);
        this->declare_parameter<int>("max_hamming",0);
        this->declare_parameter<float>("size",0.0);

        this->get_parameter<double>("camera_intrinsic.fx",fx);
        this->get_parameter<double>("camera_intrinsic.fy",fy);
        this->get_parameter<double>("camera_intrinsic.cx",cx);
        this->get_parameter<double>("camera_intrinsic.cy",cy);

        this->get_parameter<double>("camera_intrinsic.k1",k1);
        this->get_parameter<double>("camera_intrinsic.k2",k2);
        this->get_parameter<double>("camera_intrinsic.p1",p1);
        this->get_parameter<double>("camera_intrinsic.p2",p2);
        this->get_parameter<double>("camera_intrinsic.k3",k3);

        this->get_parameter("family_name",family_name);
        this->get_parameter("decimate", decimate);
        this->get_parameter("blur", blur);
        this->get_parameter("refine_edges", refine_edges);
        this->get_parameter("max_hamming", max_hamming);
        this->get_parameter("size", tag_size_mm);

        if (family_name == "tag36h11") {
            tag1.family = tag36h11_create();
        } else if (family_name == "tag25h9") {
            tag1.family = tag25h9_create();
        } else if (family_name == "tag16h5") {
            tag1.family = tag16h5_create();
        } else {
            RCLCPP_ERROR(this->get_logger(), "不支持的 tag family: %s", family_name.c_str());
            return;
        }

        // 【核心】转换为OpenCV相机内参矩阵 (3x3，双精度浮点，solvePnP最优)
        cv::Mat calib_matrix = (cv::Mat_<double>(3,3) <<
            fx,    0,    cx,   // 第一行
             0,   fy,    cy,   // 第二行
             0,    0,     1    // 第三行（固定）
        );

        // 2. 【核心】转换为 OpenCV 畸变系数矩阵 (5行1列，double)
        cv::Mat dist_coeffs = (cv::Mat_<double>(5,1) <<
            k1,   // 第1行
            k2,   // 第2行
            p1,   // 第3行
            p2,   // 第4行
            k3    // 第5行
        );

        tag1.detector = apriltag_detector_create();
        apriltag_detector_add_family(tag1.detector, tag1.family);
        tag1.detector->decimate = decimate;
        tag1.detector->blur = blur;
        tag1.detector->refine_edges = refine_edges;
        tag1.detector->max_hamming = max_hamming;
        tag1.size = tag_size_mm;


        // 1. 创建图像订阅者，订阅 RealSense 默认彩色图像话题
        // 话题说明：RealSense 相机默认发布 /color/image_raw（若有命名空间则为 /camera/color/image_raw）
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/color/image_raw",/*改为对齐后的*/                  // 订阅的话题名
            10,                                  // 消息队列大小（防止图像堆积）
            std::bind(&ColorImageSubscriber::image_callback, this, std::placeholders::_1)  // 回调函数绑定
        );

        // 2. 打印启动日志
        RCLCPP_INFO(this->get_logger(), "彩色图像订阅节点已启动！");
        RCLCPP_INFO(this->get_logger(), "订阅话题: /color/image_raw");
        RCLCPP_INFO(this->get_logger(), "操作说明：按 's' 保存当前图像 | 按 'q' 退出程序");
    }

    /**
     * @brief 析构函数：清理 OpenCV 窗口
     */
    ~ColorImageSubscriber()
    {
        cv::destroyAllWindows();
        RCLCPP_INFO(this->get_logger(), "节点已关闭，清理资源完成");
    }

private:
    /**
     * @brief 图像回调函数：处理接收到的彩色图像消息
     * @param msg 接收到的 ROS2 图像消息
     */

    struct Apriltag
    {
        apriltag_family_t* family;
        apriltag_detector_t* detector;
        double size;
        int id {};
        bool is_plan_tag {};
        Eigen::Vector3d position_corner_1 {};
        Eigen::Vector3d position_corner_2 {};
        Eigen::Vector3d position_corner_3 {};
        Eigen::Vector3d position_corner_4 {};
        Eigen::Vector3d R2_to_tag vector {};
        cv::Mat rvec {};
        cv::Mat tvec {};
    };


    double fx, fy, cx, cy;
    double k1, k2, k3, p1, p2;
    cv::Mat calib_matrix;
    cv::Mat dist_coeffs;

    std::string family_name;
    double decimate = 0, blur = 0;
    int refine_edges = 0, max_hamming = 0;
    float tag_size_mm = 0; // Tag边长（单位：毫米，需与实际打印的Tag一致）
    Apriltag tag1;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            // 将 ROS2 Image 消息转换为 OpenCV 格式（BGR8：8位BGR彩色图像，符合OpenCV标准）
            cv::Mat cv_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
			/*留空声明结果类型*/

			if (cv_image.empty()){
				RCLCPP_ERROR(this->get_logger(), "the image is empty");
				return;
			};

            cv::Mat Optimage_for_detect = imageOpt_for_Apriltag(cv::Mat &cv_image);

                    // ------------- 步骤5：检测AprilTag -------------
            // 转换图像格式（OpenCV Mat → AprilTag image_u8_t）
            image_u8_t *apriltag_image = image_u8_create(Optimage_for_detect.cols, Optimage_for_detect.rows);
            memcpy(apriltag_image->buf, Optimage_for_detect.data, Optimage_for_detect.cols * Optimage_for_detect.rows);

            // 检测Tag
            zarray_t *detections = apriltag_detector_detect(detector, apriltag_image);
            int num_tags = zarray_size(detections);
            RCLCPP_INFO(this->get_logger(), "识别到 %d 个AprilTag", num_tags);

            for (int i = 0; i < num_tags; i++)
            {
                apriltag_detection_t *detection = zarray_at(detections, i);
                int tag_id = detection->id;
                tag1.id = tag_id;
                if (tag1.id == 5)
                {
                    float tag_size_m = tag_size_mm / 1000.0f;
                    std::vector<cv::Point3f> obj_points = {
                        cv::Point3f(-tag_size_m/2, -tag_size_m/2, 0), // 左上
                        cv::Point3f( tag_size_m/2, -tag_size_m/2, 0), // 右上
                        cv::Point3f( tag_size_m/2,  tag_size_m/2, 0), // 右下
                        cv::Point3f(-tag_size_m/2,  tag_size_m/2, 0)  // 左下
                    };

                    // 提取Tag的2D角点
                    std::vector<cv::Point2f> img_points;
                    img_points.emplace_back(detection->p[0][0], detection->p[0][1]); // 左上
                    img_points.emplace_back(detection->p[1][0], detection->p[1][1]); // 右上
                    img_points.emplace_back(detection->p[2][0], detection->p[2][1]); // 右下
                    img_points.emplace_back(detection->p[3][0], detection->p[3][1]); // 左下

                    // 解算位姿（PnP + LM优化，高精度）
                    cv::solvePnP(obj_points, img_points, calib_matrix, dist_coeffs,
                                 tag_result.rvec, tag_result.tvec, false, cv::SOLVEPNP_ITERATIVE);
                    // 迭代优化位姿（提升精度）
                    cv::solvePnPRefineLM(obj_points, img_points, calib_matrix, dist_coeffs,
                                         tag_result.rvec, tag_result.tvec);
                }
            };

            // ------------- 步骤6：解算每个Tag的位姿 -------------
            // Tag四个角点的3D坐标（单位：米，Tag中心在原点，边长tag_size_mm/1000米）
            float tag_size_m = tag_size_mm / 1000.0f;
            std::vector<cv::Point3f> obj_points = {
                cv::Point3f(-tag_size_m/2, -tag_size_m/2, 0), // 左上
                cv::Point3f( tag_size_m/2, -tag_size_m/2, 0), // 右上
                cv::Point3f( tag_size_m/2,  tag_size_m/2, 0), // 右下
                cv::Point3f(-tag_size_m/2,  tag_size_m/2, 0)  // 左下
            };

            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            // 提取Tag的2D角点
            std::vector<cv::Point2f> img_points;
            img_points.emplace_back(det->p[0][0], det->p[0][1]); // 左上
            img_points.emplace_back(det->p[1][0], det->p[1][1]); // 右上
            img_points.emplace_back(det->p[2][0], det->p[2][1]); // 右下
            img_points.emplace_back(det->p[3][0], det->p[3][1]); // 左下

            // 解算位姿（PnP + LM优化，高精度）
            auto corners = img_points;
            cv::solvePnP(obj_points, img_points, calib_matrix, dist_coeffs,
                         tag1.rvec, tag1.tvec, false, cv::SOLVEPNP_ITERATIVE);
            // 迭代优化位姿（提升精度）
            cv::solvePnPRefineLM(obj_points, img_points, calib_matrix, dist_coeffs,
                                 tag1.rvec, tag1.tvec);



            // 打印位姿信息（单位：米/度）
            cv::Mat rmat;
            cv::Rodrigues(tag_result.rvec, rmat); // 旋转向量转旋转矩阵
            double roll, pitch, yaw;
            cv::RQDecomp3x3(rmat, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), roll, pitch, yaw);
            RCLCPP_INFO(this->get_logger(),
                        "Tag ID: %d | 平移(x=%.3fm, y=%.3fm, z=%.3fm) | 旋转(roll=%.2f°, pitch=%.2f°, yaw=%.2f°)",
                        det->id, tag_result.tvec.at<double>(0), tag_result.tvec.at<double>(1), tag_result.tvec.at<double>(2),
                        roll*180/CV_PI, pitch*180/CV_PI, yaw*180/CV_PI);


            // ------------- 步骤7：资源清理 -------------
            zarray_destroy(detections);
            image_u8_destroy(apriltag_image);
            apriltag_detector_destroy(detector);
            tag36h11_destroy(tf);

            // ------------- 步骤8：绘制识别结果（可选，可视化） -------------
            cv::Mat display_image = cv_image.clone();
            for (const auto& tag : results.tag_results) {
                if (!tag.detected) continue;
                // 绘制Tag四个角点
                for (const auto& corner : tag.corners) {
                    cv::circle(display_image, corner, 5, cv::Scalar(0, 255, 0), -1);
                }
                // 绘制Tag ID和中心
                cv::Point2f center(0, 0);
                for (const auto& corner : tag.corners) center += corner;
                center /= 4;
                cv::putText(display_image, "ID: " + std::to_string(tag.id), center,
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("AprilTag Detection Result", display_image);








            // 实时显示彩色图像
            cv::imshow("RealSense Color Image", cv_image);

            // 监听键盘输入（1ms 超时，避免阻塞节点）
            int key = cv::waitKey(1) & 0xFF;

            if (key == 'q')  // 按 'q' 退出程序
            {
                rclcpp::shutdown();  // 关闭 ROS2 上下文
            }
        }
        catch (cv_bridge::Exception& e)  // 捕获图像转换异常
        {
            RCLCPP_ERROR(this->get_logger(), "图像转换失败: %s", e.what());
        }
        catch (std::exception& e)  // 捕获其他通用异常
        {
            RCLCPP_ERROR(this->get_logger(), "处理图像时出错: %s", e.what());
        }
    }

    // 成员变量
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;  // 图像订阅者
    std::string save_dir_;  // 图像保存目录
    int image_count_;       // 已保存图像计数
};

/**
 * @brief 主函数：初始化 ROS2 并启动节点
 */
int main(int argc, char * argv[])
{
    // 1. 初始化 ROS2 上下文
    rclcpp::init(argc, argv);

    // 2. 创建节点实例并自旋（持续监听话题，处理回调）
    auto node = std::make_shared<ColorImageSubscriber>();
    rclcpp::spin(node);

    // 3. 关闭 ROS2 上下文（spin退出后执行）
    rclcpp::shutdown();
    return 0;
}