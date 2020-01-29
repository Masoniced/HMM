#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include "HMM.h"

using namespace std;

double inf = numeric_limits<double>::infinity();

template <typename T, typename T1>
double vector_inner_mult(vector<T> const &vec1, vector<T1> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	double sum_value = 0.0;
	if (i == j) {
		for (int n = 0; n < i; n++) {
			sum_value += (double)vec1.at(n) * vec2.at(n);
		}
	}
	else { printf("vector size not match..."); }

	return sum_value;
}

template <typename T, typename T1>
vector<double> vector_element_mult(vector<T> const &vec1, vector<T1> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	vector<double> return_vec(i);
	if (i == j) {
		for (int n = 0; n < i; n++) {
			return_vec.at(n) = (double)vec1.at(n) * vec2.at(n);
		}
	}
	else { printf("vector size not match..."); }

	return return_vec;
}

template <typename T, typename T1>
vector<double> vector_element_add(vector<T> const &vec1, vector<T1> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	vector<double> return_vec(i);
	if (i == j) {
		for (int n = 0; n < i; n++) {
			return_vec.at(n) = (double)vec1.at(n) + vec2.at(n);
		}
	}
	else { printf("vector size not match..."); }

	return return_vec;
}

template <typename T>
vector<vector<T>> matrix_element_add(vector<vector<T>> const &vec1, vector<vector<T>> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	vector<vector<T>> return_vec(i);
	if (i == j) {
		for (int n = 0; n < i; n++) {
			return_vec.at(n) = vector_element_add(vec1.at(n), vec2.at(n));
		}
	}
	else { printf("vector size not match..."); }

	return return_vec;
}

struct accum_sum_of_exp {
	//x contains the sum of exp so far, y is the next value
	double operator()(double x, double y) const {
		return x + exp(y);
	}
};

template <typename T, typename T1>
double vector_inner_mult_log(vector<T> const &vec1, vector<T1> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	double sum_value = 0.0;
	if (i == j) {
		vector<double> log_mult = vector_element_add(vec1, vec2);
		auto max_value = *max_element(log_mult.begin(), log_mult.end());
		if (max_value == -inf) {
			sum_value = -inf;
		}
		else {
			transform(log_mult.begin(), log_mult.end(), log_mult.begin(), [max_value](double d) {return d - max_value; });
			sum_value = accumulate(log_mult.begin(), log_mult.end(), 0.0, accum_sum_of_exp());
			sum_value = log(sum_value) + max_value;
		}
	}
	else { printf("vector size not match..."); }

	return sum_value;
}

template <typename T, typename T1>
vector<double> vector_element_add_log(vector<T> const &vec1, vector<T1> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	vector<double> return_vec(i);
	if (i == j) {
		for (int n = 0; n < i; n++) {
			double temp = exp(vec1.at(n)) + exp(vec2.at(n));
			return_vec.at(n) = log(temp);
		}
	}
	else { printf("vector size not match..."); }

	return return_vec;
}

template <typename T>
vector<vector<T>> matrix_element_add_log(vector<vector<T>> const &vec1, vector<vector<T>> const &vec2) {
	int i = vec1.size();
	int j = vec2.size();
	vector<vector<T>> return_vec(i);
	if (i == j) {
		for (int n = 0; n < i; n++) {
			return_vec.at(n) = vector_element_add_log(vec1.at(n), vec2.at(n));
		}
	}
	else { printf("vector size not match..."); }

	return return_vec;
}

template <typename T>
vector<T> get_column(vector<vector<T>> const &vec, int n) {
	int length = vec.size();
	vector<double> temp(length);
	for (int i = 0; i < length; i++) {
		temp.at(i) = vec.at(i).at(n);
	}

	return temp;
}

vector<double> rand_initial(int n) {
	int length = n;
	random_device rd;
	mt19937 e2;
	e2.seed(rd());
	uniform_real_distribution<> dist(0, 1);
	vector<double> temp(length, 0);
	for (int i = 0; i < n; i++) {
		double sum_temp;
		sum_temp = accumulate(temp.begin(), temp.end(), 0.0);
		if (i != n - 1) {
			double r = dist(e2);
			temp.at(i) = (1.0 - sum_temp) * r;
		}
		else {
			temp.at(i) = sum_temp;
		}
	}
	return temp;
}

void HMM::para_ini_decode(void) {
	num_stat = A_trans.size();
	num_obs = B_trans.size();
	transform(initial_prob.begin(), initial_prob.end(), initial_prob.begin(), [](double d) {return log(d); });
	for_each(A_trans.begin(), A_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return log(d); }); }
	);
	for_each(B_trans.begin(), B_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return log(d); }); }
	);

	ob_seq_length = observation_seq.size();
	verti.assign(ob_seq_length, vector<double>(num_stat));
	back_path.assign(num_stat, vector<int>(ob_seq_length));
	verti.at(0) = vector_element_add(initial_prob, B_trans.at(observation_seq.at(0)));
	for (int i = 0; i < num_stat; i++) {
		back_path.at(i).at(0) = i;
	}
}

void HMM::decode(void) {
	para_ini_decode();
	for (int i = 1; i < ob_seq_length; i++) {
		for (int j = 0; j < num_stat; j++) {
			vector<double> column_A_trans = get_column(A_trans, j);
			vector<double> temp_vec = vector_element_add(column_A_trans, verti.at(i - 1));
			auto it = max_element(temp_vec.begin(), temp_vec.end());
			int temp_index = it - temp_vec.begin();
			verti.at(i).at(j) = temp_vec.at(temp_index) + B_trans.at(observation_seq.at(i)).at(j);
			back_path.at(j) = back_path.at(temp_index);
			back_path.at(j).at(i) = temp_index;
		}
	}

	auto final_it = max_element(verti.at(num_obs - 1).begin(), verti.at(num_obs - 1).end());
	int index = final_it - verti.at(num_obs - 1).begin();
	verti_decode = back_path.at(index);
}

void HMM::para_ini_inference(void) {
	ob_seq_length = observation_seq.size();
	auto it_obs = max_element(observation_seq.begin(), observation_seq.end());
	int max_obs_index = it_obs - observation_seq.begin();
	num_obs = observation_seq.at(max_obs_index) + 1;
	last_log_obs = 0.0;

	A_trans.assign(num_stat, vector<double>(num_stat));
	for (int i = 0; i < num_stat; i++) {
		A_trans.at(i) = rand_initial(num_stat);
	}
	B_trans.assign(num_obs, vector<double>(num_stat));
	for (int i = 0; i < num_stat; i++) {
		vector<double> temp = rand_initial(num_obs);
		for (int j = 0; j < num_obs; j++) {
			B_trans.at(j).at(i) = temp.at(j);
		}
	}

	a_forward.assign(ob_seq_length, vector<double>(num_stat));
	b_backward.assign(ob_seq_length, vector<double>(num_stat));
	initial_prob.assign(num_stat, 0);
	initial_prob = rand_initial(num_stat);

	eps_obs.assign(ob_seq_length - 1, vector<vector<double>>(num_stat, vector<double>(num_stat)));
	obs_prob.assign(ob_seq_length - 1, 0);
	gamma_obs.assign(ob_seq_length, vector<double>(num_stat));
	for (int ob_it = 0; ob_it < ob_seq_length; ob_it++) {
		if (obs_distribution.find(observation_seq.at(ob_it)) == obs_distribution.end()) {
			vector<int> temp;
			obs_distribution.insert(unordered_map<int, vector<int>>::value_type(observation_seq.at(ob_it), temp));
		}
		obs_distribution.at(observation_seq.at(ob_it)).push_back(ob_it);
	}
	//Log transform
	transform(initial_prob.begin(), initial_prob.end(), initial_prob.begin(), [](double d) {return log(d); });
	for_each(A_trans.begin(), A_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return log(d); }); }
	);
	for_each(B_trans.begin(), B_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return log(d); }); }
	);
}

void HMM::inference_expectation(void) {
	a_forward.at(0) = vector_element_add(initial_prob, B_trans.at(observation_seq.at(0)));
	vector<double> ini_b(num_stat, 0);
	b_backward.at(ob_seq_length - 1) = ini_b;

	for (int ob_it = 1; ob_it < ob_seq_length; ob_it++) {
		for (int stat_it = 0; stat_it < num_stat; stat_it++) {
			vector<double> column_A_trans = get_column(A_trans, stat_it);
			double temp_a = vector_inner_mult_log(a_forward.at(ob_it - 1), column_A_trans);
			a_forward.at(ob_it).at(stat_it) = temp_a + B_trans.at(observation_seq.at(ob_it)).at(stat_it);
		}
	}

	for (int ob_it = ob_seq_length - 2; ob_it > -1; ob_it--) {
		for (int stat_it = 0; stat_it < num_stat; stat_it++) {
			vector<double> temp_b;
			temp_b = vector_element_add(b_backward.at(ob_it + 1), B_trans.at(ob_it + 1));
			b_backward.at(ob_it).at(stat_it) = vector_inner_mult_log(temp_b, A_trans.at(stat_it));
		}
	}

	for (int ob_it = 0; ob_it < ob_seq_length - 1; ob_it++) {
		obs_prob.at(ob_it) = vector_inner_mult_log(a_forward.at(ob_it), b_backward.at(ob_it));
		for (int stat_it_c = 0; stat_it_c < num_stat; stat_it_c++) {
			for (int stat_it_n = 0; stat_it_n < num_stat; stat_it_n++) {
				eps_obs.at(ob_it).at(stat_it_c).at(stat_it_n) = (A_trans.at(stat_it_c).at(stat_it_n) + B_trans.at(observation_seq.at(ob_it + 1)).at(stat_it_n)\
					+ a_forward.at(ob_it).at(stat_it_c) + b_backward.at(ob_it + 1).at(stat_it_n)) - obs_prob.at(ob_it);
			}
			gamma_obs.at(ob_it).at(stat_it_c) = a_forward.at(ob_it).at(stat_it_c) + b_backward.at(ob_it).at(stat_it_c) - obs_prob.at(ob_it);
			if (ob_it == ob_seq_length - 2) {
				gamma_obs.at(ob_it + 1).at(stat_it_c) = a_forward.at(ob_it + 1).at(stat_it_c) + b_backward.at(ob_it + 1).at(stat_it_c) - obs_prob.at(ob_it);
			}
		}
	}
}

void HMM::inference_maximization(void) {
	vector<vector<double>> est_B_trans(num_obs, vector<double>(num_stat));
	for (auto it = obs_distribution.begin(); it != obs_distribution.end(); it++) {
		vector<double> temp(num_stat, log(0.0));
		int temp_length = it->second.size();
		for (int i = 0; i < temp_length; i++) {
			temp = vector_element_add_log(temp, gamma_obs.at(it->second.at(i)));
		}
		est_B_trans.at(it->first) = temp;
	}

	vector<double> sum_gamma(num_stat, 0.0);
	for (int i = 0; i < num_stat; i++) {
		vector<double> column_est_B_trans = get_column(est_B_trans, i);
		auto max_value = *max_element(column_est_B_trans.begin(), column_est_B_trans.end());
		transform(column_est_B_trans.begin(), column_est_B_trans.end(), column_est_B_trans.begin(), [max_value](double d) {return d - max_value; });
		double sum_column = accumulate(column_est_B_trans.begin(), column_est_B_trans.end(), 0.0, accum_sum_of_exp());
		sum_gamma.at(i) = log(sum_column) + max_value;
	}
	for_each(est_B_trans.begin(), est_B_trans.end(), [&](vector<double> &row) {
		transform(row.begin(), row.end(), sum_gamma.begin(), row.begin(), [](double d1, double d2) {return d1 - d2; }); 
	});
	B_trans = est_B_trans;
	double total_gamma_0 = accumulate(gamma_obs.at(0).begin(), gamma_obs.at(0).end(), 0.0, accum_sum_of_exp());
	vector<double> new_initial_prob = gamma_obs.at(0);
	transform(new_initial_prob.begin(), new_initial_prob.end(), new_initial_prob.begin(), [total_gamma_0](double d) {return d - log(total_gamma_0); });
	initial_prob = new_initial_prob;
	vector<vector<double>> sum_eps_obs(num_stat, vector<double>(num_stat, log(0.0)));
	vector<double> sum_eps(num_stat, 0.0);

	for (int i = 0; i < num_stat; i++) {
		sum_eps_obs = matrix_element_add_log(sum_eps_obs, eps_obs.at(i));
	}
	for (int i = 0; i < num_stat; i++) {
		double temp = accumulate(sum_eps_obs.at(i).begin(), sum_eps_obs.at(i).end(), 0.0, accum_sum_of_exp());
		sum_eps.at(i) = log(temp);
	}
	for (int i = 0; i < num_stat; i++) {
		double temp = sum_eps.at(i);
		transform(sum_eps_obs.at(i).begin(), sum_eps_obs.at(i).end(), sum_eps_obs.at(i).begin(), [temp](double d) {return d - temp; });
	}
	A_trans = sum_eps_obs;
}

void HMM::check_converge(void) {
	current_log_obs = accumulate(obs_prob.begin(), obs_prob.end(), 0.0, accum_sum_of_exp());
	current_log_obs = log(current_log_obs) / (ob_seq_length - 1);
	double diff = fabs(current_log_obs - last_log_obs);

	if (diff > tol) {
		last_log_obs = current_log_obs;
	}
	else
	{
		end_check = true;
	}
}

void HMM::inference(void) {
	para_ini_inference();

	for (int i = 0; i < iter_limit; i++) {
		inference_expectation();
		inference_maximization();
		check_converge();
		num_iter += 1;
		if (end_check) {
			break;
		}
	}

	Final_initial_prob = initial_prob;
	Final_A_trans = A_trans;
	Final_B_trans = B_trans;
	transform(Final_initial_prob.begin(), Final_initial_prob.end(), Final_initial_prob.begin(), [](double d) {return exp(d); });
	for_each(Final_A_trans.begin(), Final_A_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return exp(d); }); }
	);
	for_each(Final_B_trans.begin(), Final_B_trans.end(),
		[&](vector<double> &row) {transform(row.begin(), row.end(), row.begin(), [](double d) {return exp(d); }); }
	);
}
