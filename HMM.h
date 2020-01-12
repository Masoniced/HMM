#ifndef _HMM_H
#define _HMM_H

#include <vector>
#include <unordered_map>
#include <limits>

class HMM
{
private:
	//A_trans has row like state transformation matrix; A_trans[i][j] is the prob from i->j
	//B_trans has the column like observation prob distribution; B_trans[i][j] is the prob of state i to obseration j
	std::vector<double> initial_prob; //Initial Probability pi
	std::vector<int> observation_seq; // Observation seqeunce
	std::vector<int> state_seq;
	std::vector<std::vector<double>> A_trans;
	std::vector<std::vector<double>> B_trans;
	std::vector<std::vector<double>> a_forward;
	std::vector<std::vector<double>> b_backward;
	std::vector<std::vector<std::vector<double>>> eps_obs;
	std::vector<std::vector<double>> gamma_obs;
	std::vector<double> obs_prob;
	std::unordered_map<int, std::vector<int>> obs_distribution;

	int num_obs;
	int num_stat;
	int ob_seq_length;
	int iter_limit;
	double last_log_obs;
	double current_log_obs;
	bool end_check = false;
	double tol = 1e-9;
	int num_iter = 0;

	std::vector<std::vector<double>> verti;
	std::vector<std::vector<int>> back_path;

public:
	std::vector<int> verti_decode;
	std::vector<std::vector<double>> Final_A_trans;
	std::vector<std::vector<double>> Final_B_trans;
	std::vector<double> Final_initial_prob;
	//Vertibi for coding and decoding process
	void para_ini_decode(void);
	void decode(void);
	//EM for learning process
	void para_ini_inference(void);
	void inference_expectation(void);
	void inference_maximization(void);
	void check_converge(void);
	void inference(void);

	HMM(std::vector<double> INITIAL_PROB, std::vector<int> OBSERVATION_SEQ, std::vector<std::vector<double>> A_TRANS, std::vector<std::vector<double>> B_TRANS) :\
		initial_prob(INITIAL_PROB), observation_seq(OBSERVATION_SEQ), A_trans(A_TRANS), B_trans(B_TRANS) {};

	HMM(std::vector<int> OBSERVATION_SEQ, int NUM_STAT, int ITER_LIMIT, int TOL) : observation_seq(OBSERVATION_SEQ), num_stat(NUM_STAT), iter_limit(ITER_LIMIT), tol(TOL) {};

	HMM(std::vector<int> OBSERVATION_SEQ, int NUM_STAT, int ITER_LIMIT) : observation_seq(OBSERVATION_SEQ), num_stat(NUM_STAT), iter_limit(ITER_LIMIT) {};

};

#endif
