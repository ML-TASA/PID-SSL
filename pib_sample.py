import torch
import numpy as np
import logging
from scipy.spatial.distance import jensenshannon

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SamplingMechanism:
    def __init__(self, model, distance_metric=jensenshannon):
        """
        Initialize the SamplingMechanism class.

        Args:
        - model (torch.nn.Module): The model which should include the functionality to compute latent variables and balancing scores.
        - distance_metric (function): The metric used to calculate the distance between scores, default is JS divergence.
        """
        self.model = model
        self.distance_metric = distance_metric

    def compute_balancing_score(self, s, lambda_e):
        """
        Compute the balancing score for the latent variable 's'.

        Args:
        - s (torch.Tensor): Latent variable.
        - lambda_e (torch.Tensor): The learning balancing function for the batch.

        Returns:
        - ba_s (torch.Tensor): The balancing score.
        """
        try:
            # Compute the balancing score using the model
            ba_s = self.model.compute_balancing_score(s, lambda_e)
            return ba_s
        except Exception as e:
            logging.error(f"Balancing score computation failed: {e}")
            raise

    def compute_js_divergence(self, p, q):
        """
        Compute Jensen-Shannon Divergence (JS divergence).

        Args:
        - p (numpy.ndarray): First probability distribution.
        - q (numpy.ndarray): Second probability distribution.

        Returns:
        - js_div (float): The JS divergence between the distributions.
        """
        try:
            return self.distance_metric(p, q)
        except Exception as e:
            logging.error(f"JS divergence computation failed: {e}")
            raise

    def mini_batch_sampling(self, Xtr, a, batch_size=64):
        """
        Mini-batch sampling strategy based on balancing score matching.

        Args:
        - Xtr (list): The training dataset, consisting of (x+, xlabel) pairs.
        - a (int): The number of samples to be sampled.
        - batch_size (int): The size of the batch.

        Returns:
        - mini_batch (list): The sampled mini-batch.
        """
        mini_batch = []
        sampled_indices = set()

        try:
            first_sample_idx = np.random.choice(len(Xtr))
            x_plus, xlabel = Xtr[first_sample_idx]
            s = self.model.compute_latent_variable(x_plus, xlabel)  # Compute the latent variable
            ba_s = self.compute_balancing_score(s, self.model.lambda_e)  # Compute the balancing score
            mini_batch.append((x_plus, xlabel, ba_s))
            sampled_indices.add(first_sample_idx)
        except Exception as e:
            logging.error(f"First sample failed: {e}")
            raise

        for i in range(1, a + 1):
            min_distance = float('inf')
            best_match_idx = None

            # For each sample, calculate the JS divergence and find the most similar sample
            for j in range(len(Xtr)):
                if j in sampled_indices:  # Ensure no repeated sampling
                    continue
                x_plus_j, xlabel_j = Xtr[j]
                s_j = self.model.compute_latent_variable(x_plus_j, xlabel_j)
                ba_s_j = self.compute_balancing_score(s_j, self.model.lambda_e)

                # Calculate JS divergence
                try:
                    distance = self.compute_js_divergence(mini_batch[i-1][2], ba_s_j)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_idx = j
                except Exception as e:
                    logging.error(f"JS divergence calculation failed: {e}")
                    continue  

            if best_match_idx is not None:
                x_plus_best, xlabel_best = Xtr[best_match_idx]
                s_best = self.model.compute_latent_variable(x_plus_best, xlabel_best)
                ba_s_best = self.compute_balancing_score(s_best, self.model.lambda_e)
                mini_batch.append((x_plus_best, xlabel_best, ba_s_best))
                sampled_indices.add(best_match_idx)
            else:
                logging.warning(f"Matching for sample {i+1} failed, stopping sampling.")
                break

        # Check if sampling was successful
        if len(mini_batch) < a + 1:
            logging.warning(f"Mini-batch sampling did not reach the expected size: {len(mini_batch)}. This could be due to uneven data distribution.")
        
        return mini_batch


def generate_mini_batch(Xtr, model, a, batch_size=64):
    """
    External interface function to generate mini-batch.

    Args:
    - Xtr (list): The training dataset.
    - model (torch.nn.Module): The trained model.
    - a (int): The mini-batch size.
    - batch_size (int): The size of the batch.

    Returns:
    - mini_batch (list): The generated mini-batch.
    """
    sampling_mechanism = SamplingMechanism(model)
    mini_batch = sampling_mechanism.mini_batch_sampling(Xtr, a, batch_size)
    return mini_batch
