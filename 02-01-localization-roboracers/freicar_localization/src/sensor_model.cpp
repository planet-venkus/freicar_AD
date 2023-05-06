/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

#include "sensor_model.h"
#include "ros/ros.h"
#include "visualization_msgs/MarkerArray.h"
#include "ros_vis.h"


std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

/*
 * Returns k random indeces between 0 and N
 */
std::vector<int> pick(int N, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_set<int> elems = pickSet(N, k, gen);

    std::vector<int> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

/*
 * Constructor of sensor model. Builds KD-tree indeces
 */
sensor_model::sensor_model(PointCloud<float> map_data, std::map<std::string, PointCloud<float> > sign_data, std::shared_ptr<ros_vis> visualizer, bool use_lane_reg):map_data_(map_data), sign_data_(sign_data), map_index_(2 /*dim*/, map_data_, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ), visualizer_(visualizer)
{
    // Creating KD Tree indeces for fast nearest neighbor search
    map_index_.buildIndex();
    for(auto ds = sign_data_.begin(); ds != sign_data_.end(); ds++){
        std::cout << "Creating kd tree for sign type: " << ds->first << " with " << ds->second.pts.size() << " elements..." << std::endl;
        sign_indeces_[ds->first] = std::unique_ptr<map_kd_tree>(new map_kd_tree(2, ds->second, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
        sign_indeces_[ds->first]->buildIndex();
    }
    use_lane_reg_ = use_lane_reg;
}

/*
 * For any given observed lane center points return the nearest lane-center points in the map
 * This is using a KD-Tree for efficient match retrieval
 */
std::vector<Eigen::Vector3f> sensor_model::getNearestPoints(std::vector<Eigen::Vector3f> sampled_points){
    // Get data association
    assert(map_data_.pts.size() > 0);
    std::vector<Eigen::Vector3f> corr_map_associations;
    for(size_t i=0; i < sampled_points.size(); i++){
        // search nearest neighbor for sampled point in map
        float query_pt[2] = { static_cast<float>(sampled_points.at(i).x()), static_cast<float>(sampled_points.at(i).y())};

        const size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr );
        map_index_.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // Gather map vector
        Point_KD<float> corr_p = map_data_.pts.at(ret_index);
        corr_map_associations.push_back(Eigen::Vector3f((corr_p.x), (corr_p.y), 0.0));
    }
    return corr_map_associations;
}

/*
 * For given observed signs return nearest sign positions with the same type.
 * This is using a KD-Tree for efficient match retrieval.
 * Returns a empty vector if not possible
 */
std::vector<Eigen::Vector3f> sensor_model::getNearestPoints(std::vector<Sign> observed_signs){
    // Get data association
    std::vector<Eigen::Vector3f> corr_map_associations;
    for(size_t i=0; i < observed_signs.size(); i++){
        const Sign& s = observed_signs.at(i);
        if(sign_indeces_.find(s.type) != sign_indeces_.end()){
            // search nearest neighbor for sampled point in map
            float query_pt[2] = {s.position[0], s.position[1]};

            const size_t num_results = 1;
            size_t ret_index;
            float out_dist_sqr;
            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_index, &out_dist_sqr );
            sign_indeces_[s.type]->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

            // Gather sign position from map
            if(out_dist_sqr < 1e30){
                Point_KD<float> corr_p = sign_data_[s.type].pts.at(ret_index);
                corr_map_associations.push_back(Eigen::Vector3f((corr_p.x), (corr_p.y), 0.0));
            }else{
                std::cerr << "Invalid query..." << std::endl;
                return std::vector<Eigen::Vector3f>();
            }
        }else{
            std::cerr << "No corrensponding sign in map kd indeces..." << std::endl;
            return std::vector<Eigen::Vector3f>();
        }
    }
    return corr_map_associations;
}

/*
 * Transforms a given list of 3D points by a given affine transformation matrix
 */
std::vector<Eigen::Vector3f> sensor_model::transformPoints(const std::vector<Eigen::Vector3f> points, const Eigen::Transform<float,3,Eigen::Affine> transform){
    std::vector<Eigen::Vector3f> transformed;
    for(size_t i =0; i < points.size(); i++){
        Eigen::Vector3f p_world = transform * points.at(i);
        transformed.push_back(p_world);
    }
    return transformed;
}

/*
 * Returns sum of given float-vector
 */
float sensor_model::sumWeights(const std::vector<float>& weights){
    float sum = 0.0f;
    for(auto i = weights.begin(); i != weights.end(); i++){
        sum += *i;
    }
    return sum;
}

float probability_density_function(float mean, float stddev, float pos){
    float variance = stddev * stddev;
    return 1./(sqrt(2.*M_PI*variance)) * exp(-((std::pow(pos-mean, 2.))/(2.*variance)));
}

float sensor_model::SignMeasurementPoseProbabilityNearest(const std::vector<Sign>& observed_signs, const std::vector<Eigen::Vector3f>& data_associations_signs){
    size_t N = observed_signs.size();

    float error_signs = 0.0f;
    // Observed Signs #####################################
    for(size_t i=0; i < observed_signs.size(); i++){
        const Eigen::Vector3f& p = observed_signs.at(i).position;
        const Eigen::Vector3f& q = data_associations_signs.at(i);
        error_signs +=  std::sqrt((q-p).squaredNorm());
    }

    error_signs /= static_cast<float>(N);
    return probability_density_function(0, 0.4, error_signs);
//    return std::max(std::min(double(exp(-error_signs)), 1.0), 0.1);
}

float sensor_model::SignMeasurementPoseProbability(const std::vector<Sign>& observed_signs_map, Eigen::Transform<float,3,Eigen::Affine> particle_transform){
    float prob = 0.0f;
    float norm_val = 0;
    // Observed Signs #####################################
    for(size_t i=0; i < observed_signs_map.size(); i++){
        for (size_t k=0; k < sign_data_[observed_signs_map.at(i).type].pts.size(); k++) {

            Point_KD<float> corr_p = sign_data_[observed_signs_map.at(i).type].pts.at(k);
            const Eigen::Vector3f &p = observed_signs_map.at(i).position;
            const Eigen::Vector3f &q = Eigen::Vector3f(corr_p.x, corr_p.y, 0.);

            Eigen::Vector3f q_in_p_pose = particle_transform.inverse() * q;
            float angle = abs(atan2(q_in_p_pose.y(), q_in_p_pose.x()) * (180.0 / M_PI));
            if (angle < 85. / 2) {   // This condition is essentially a prior (setting those priors to
                // 0 that do not fall in the condition)
                float error_sign = std::sqrt((q - p).squaredNorm());
                // exp(-error_sign) is another prior that says the further the signs are away, the less likely they get
                // detected
                prob += exp(-error_sign) * probability_density_function(0, 0.3, error_sign);
                norm_val += exp(-error_sign);
            }
        }
    }
    if(norm_val > 0) {
        prob /= norm_val;
    }

    return prob;
}

float sensor_model::LaneMeasurementPoseProbability(const std::vector<Eigen::Vector3f>& observed_points, const std::vector<Eigen::Vector3f>& data_associations, const std::vector<float>& weights, const float total_weight){

    float prob = 0.0f;
    float norm_val = 0.0f;
    // Observed Lane Points ###############################
    size_t i=0;
    for(i=0; i < data_associations.size(); i++){
        const Eigen::Vector3f& p = observed_points.at(i);
        const Eigen::Vector3f& q = data_associations.at(i);
        float error = std::sqrt((q-p).squaredNorm());
        prob += 1 * probability_density_function(0, 0.5, error);
//        prob += probability_density_function(0, 0.2, error);
//        norm_val += weight;
    }
    prob /= data_associations.size();
    return prob;
}

/*
 * Transforms sign position by a given affine transformation matrix
 */
std::vector<Sign> sensor_model::transformSigns(const std::vector<Sign>& signs, const Eigen::Transform<float,3,Eigen::Affine>& particle_pose){
    std::vector<Sign> transformed_signs;
    for(size_t i =0; i < signs.size(); i++){
        const Sign& s = signs.at(i);
        Sign t_s = s;
        t_s.position = particle_pose * s.position;
        transformed_signs.push_back(t_s);
    }
    return transformed_signs;
}



/*
 * ##########################IMPLEMENT ME###############################################################################
 * Sensor-model. This function does the following:
 * --Calculates the likelihood of every particle being at its respective pose.
 * The likelihood should be stored in the particles weight member variable
 * The observed_signs variable contain all observed signs at the current timestep. They are relative to freicar_X/base_link.
 * The current particles are given with the variable "particles"
 * The true positions of all signs for a given type are stored in: sign_data_[observed_signs.at(i).type].pts , where
 * observed_signs.at(i).type is the sign_type of the i'th observed sign and pts is a array of positions (that have
 * the member x and y)
 * For lane regression data: The function getNearestPoints() might come in handy for getting the closest points to the
 * sampled and observed lane center points.
 *
 * The variable max_prob must be filled with the highest likelihood among all particles. If the average
 * of the last BEST_PARTICLE_HISTORY (defined in particle_filter.h) max_prob values is under the value
 * QUALITY_RELOC_THRESH (defined in particle_filter.h) a resampling will be initiated. So you may want to adjust the threshold.
 *
 * The function needs to return True if everything was successfull and False otherwise.

 */
bool sensor_model::calculatePoseProbability(const std::vector<cv::Mat> lane_regression, const std::vector<Sign> observed_signs, std::vector<Particle>& particles, float& max_prob){
    // Check if there is a bev lane regression matrix available. If so, use it in the observation step

    bool success = false;
    Particle particle;

//    std::cout << "############### Inside calculate posEEEE probability ####################" << '\n';
    // ###################### Sign Model ###########################################
    if(lane_regression.size() == 0){
        for(int i = 0; i < particles.size(); i++){
            particle = particles[i];
            std::vector<Sign> obs_signs_particle = transformSigns(observed_signs, particle.transform);
            float pb = SignMeasurementPoseProbability(obs_signs_particle, particle.transform);
            if(pb > max_prob){
                max_prob = pb;
            }
            particles[i].weight = pb;
        }


        success = true;
    return success;

    }

    // ######################################## If Lane regression ########################################
    else if(lane_regression.size() > 0){

        std::vector<Eigen::Vector3f> sampled_lreg;
        std::vector<float> weights;
        useWeightsfFromLaneRegression(lane_regression, sampled_lreg, weights); //function to sample pixel values from lane reg birds eye image
        for (int i = 0; i < particles.size(); i++) {
            // Sample points from the image
            particle = particles[i];
            float avg_weight;
            float pb;
            //weight calculations for signs
            std::vector<Sign> obs_signs_particle = transformSigns(observed_signs, particle.transform);
            pb = SignMeasurementPoseProbability(obs_signs_particle, particle.transform);
            avg_weight = pb;
            if (pb > max_prob) {
                max_prob = pb;
            }
            //weight calculations when some meaningful weights were yielded by Lane regression
            if(sampled_lreg.size() != 0 ) {
                std::vector<Eigen::Vector3f> transformed_points = transformPoints(sampled_lreg, particle.transform);
                std::vector<Eigen::Vector3f> corr_map_associations = sensor_model::getNearestPoints(transformed_points);
                float particle_prob = sensor_model::LaneMeasurementPoseProbability(transformed_points,
                                                                                   corr_map_associations,
                                                                                   weights, 1.0);
                if (particle.weight != 0) {
                    avg_weight = 0.7*pb + 0.3*particle_prob;
//                    avg_weight = particle_prob;

                }
            }
            particles[i].weight = avg_weight;
        }
        success = true;
        return success;

    }
}

void sensor_model::useWeightsfFromLaneRegression(const std::vector<cv::Mat> &lane_regression,
                                                 std::vector<Eigen::Vector3f> &sampled_lreg,
                                                 std::vector<float> &weights) const {
    cv::Mat img = lane_regression[0];//To extract pixels with reasonable grayscale pixel values
//    cv::namedWindow("Lreg Birds eye view");
//    cv::imshow("Lreg Birds eye view", img);
//    cv::waitKey(1);
    cv::threshold(img, img, REG_THRESH, 255, cv::THRESH_BINARY);
    cv::Mat locations;
    //To extract locations of pixels with reasonable grayscale pixel values
    cv::findNonZero(img, locations);
    int num_max_sample = locations.rows;
    const int sample_num = NUM_SAMPLES;
    if (num_max_sample > sample_num) {
        //To sample from the above extracted pixels
        std::vector<int> samples = pick(num_max_sample, sample_num);
        for (size_t i = 0; i < samples.size(); i++) {
            // pnt is in cm
            cv::Point pnt = locations.at<cv::Point>(samples.at(i));
            float x_m = pnt.y / 100.0f;
            float y_m = pnt.x / 100.0f - img.cols / 100.0f;
            int cell = (int) img.at<uchar>(pnt.x, pnt.y);
            weights.push_back(cell);
            sampled_lreg.push_back(Eigen::Vector3f(x_m, y_m, 0));
        }
    }
}






