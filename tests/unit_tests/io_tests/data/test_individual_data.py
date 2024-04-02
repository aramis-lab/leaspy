import pytest

from leaspy.exceptions import LeaspyDataInputError
from leaspy.io.data.individual_data import IndividualData

from tests import LeaspyTestCase


class IndividualDataTest(LeaspyTestCase):

    def test_constructor(self):
        data_int = IndividualData(1)
        self.assertEqual(data_int.idx, 1)
        self.assertEqual(data_int.timepoints, None)
        self.assertEqual(data_int.observations, None)
        self.assertEqual(data_int.covariates, {})

        data_float = IndividualData(1.2)
        self.assertEqual(data_float.idx, 1.2)

        data_string = IndividualData('test')
        self.assertEqual(data_string.idx, 'test')

    def test_add_observations(self):
        # Add first observation
        data = IndividualData('test')
        data.add_observations([70], [[30]])

        self.assertEqual(data.idx, 'test')
        self.assertEqual(data.timepoints.tolist(), [70])
        self.assertEqual(data.observations.tolist(), [[30]])

        # Add new observations
        data.add_observations([80, 75], [[40], [35]])
        self.assertEqual(data.timepoints.tolist(), [70, 75, 80])
        self.assertEqual(data.observations.tolist(), [[30], [35], [40]])

        with pytest.raises(LeaspyDataInputError):
            data.add_observations([70], [[40]])

    def test_add_covariates(self):
        data = IndividualData("test")
        covariates_dict = {
            "gender": "male",
            "weight": 60
        }
        data.add_covariates(covariates_dict)
        self.assertEqual(data.covariates, covariates_dict)

        with pytest.raises(TypeError):
            data.add_covariates({5: 5})
        
        with pytest.raises(LeaspyDataInputError):
            data.add_covariates({"weight": 5})
        
    def test_add_event(self):
        # Add first observation
        data = IndividualData('test')
        data.add_event(2., True)

        self.assertEqual(data.event_time, 2.)
        self.assertEqual(data.event_bool, True)
