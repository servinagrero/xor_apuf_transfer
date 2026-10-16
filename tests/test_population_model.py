import unittest

import numpy as np

from puf_transfer import (
    CorrelatedXORArbiterPUF,
    generate_weights_bias,
    random_crps,
)
from puf_transfer.noise import calibrate_noisiness, initial_noisiness_for_ber


class PopulationModelTests(unittest.TestCase):
    def test_beta_endpoints_and_xor_parity(self):
        challenges = random_crps(256, 16, seed=1)
        for k in (2, 3):
            shared_weights, shared_bias = generate_weights_bias(
                16, k=k, beta=1.0, seed=2
            )
            np.testing.assert_allclose(
                shared_weights,
                np.repeat(shared_weights[:1], k, axis=0),
            )
            puf = CorrelatedXORArbiterPUF(
                16, k, shared_weights, shared_bias, alpha=1.0, seed=3
            )
            response = np.sign(puf.val(challenges))
            if k % 2 == 0:
                np.testing.assert_array_equal(response, np.ones(len(challenges)))
            else:
                np.testing.assert_array_equal(
                    response, np.sign(puf.chain(0).val(challenges))
                )

    def test_alpha_endpoints(self):
        shared_weights, shared_bias = generate_weights_bias(
            16, k=2, beta=0.5, seed=4
        )
        identical_a = CorrelatedXORArbiterPUF(
            16, 2, shared_weights, shared_bias, alpha=1.0, seed=5
        )
        identical_b = CorrelatedXORArbiterPUF(
            16, 2, shared_weights, shared_bias, alpha=1.0, seed=6
        )
        np.testing.assert_allclose(identical_a.weight_array, identical_b.weight_array)

        independent_a = CorrelatedXORArbiterPUF(
            16, 2, shared_weights, shared_bias, alpha=0.0, seed=5
        )
        independent_b = CorrelatedXORArbiterPUF(
            16, 2, shared_weights, shared_bias, alpha=0.0, seed=6
        )
        self.assertFalse(
            np.allclose(independent_a.weight_array, independent_b.weight_array)
        )

    def test_output_ber_calibration(self):
        challenges = random_crps(10_000, 16, seed=7)
        shared_weights, shared_bias = generate_weights_bias(
            16, k=3, beta=0.8, seed=8
        )

        def factory(noisiness):
            return CorrelatedXORArbiterPUF(
                16,
                3,
                shared_weights,
                shared_bias,
                alpha=0.9,
                seed=9,
                noisiness=noisiness,
            )

        noisiness, achieved = calibrate_noisiness(
            factory, challenges, 0.05, iterations=10
        )
        self.assertGreater(noisiness, 0)
        self.assertAlmostEqual(achieved, 0.05, delta=0.01)
        self.assertGreater(initial_noisiness_for_ber(3, 0.05), 0)


if __name__ == "__main__":
    unittest.main()
