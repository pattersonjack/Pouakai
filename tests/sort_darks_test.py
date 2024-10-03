import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from pouakai.sort_images import dark_info_grab, sort_darks


class TestDarkInfoGrab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Common mock data for all test cases."""
        cls.filename = 'dark_2024'
        cls.filepath = f'path/to/{cls.filename}.fits'
        cls.telescope_name = 'Test Telescope'
        cls.exp_time = 128
        cls.jd = 2459205.5
        cls.date = '2024-10-10'

    @patch('builtins.print')
    @patch('astropy.io.fits.open')
    def test_dark_info_grab_valid_file(self, mock_fits_open, mock_print):
        mock_fits_header = {
            'TELESCOP': self.telescope_name,
            'EXPTIME': self.exp_time,
            'JD': self.jd,
            'DATE-OBS': self.date
        }
        mock_fits_open.return_value = [MagicMock(header=mock_fits_header)]

        result_df = dark_info_grab(self.filepath, verbose=False)

        expected_data = {
            'name': [self.filename],
            'telescope': [self.telescope_name],
            'exptime': [self.exp_time],
            'jd': [self.jd],
            'date': [self.date],
            'filename': [self.filepath]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result_df, expected_df)

        mock_print.assert_not_called()

    @patch('builtins.print')
    @patch('astropy.io.fits.open')
    def test_dark_info_grab_missing_telescope_header(self, mock_fits_open, mock_print):
        mock_header = {
            'TELESCOP': '',
            'EXPTIME': self.exp_time,
            'JD': self.jd,
            'DATE-OBS': self.date
        }
        mock_fits_open.return_value = [MagicMock(header=mock_header)]

        result_df = dark_info_grab(self.filepath, verbose=False)

        expected_data = {
            'name': [self.filename],
            'telescope': ['B&C'],  # Default value
            'exptime': [self.exp_time],
            'jd': [self.jd],
            'date': [self.date],
            'filename': [self.filepath]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result_df, expected_df)

        mock_print.assert_not_called()

    @patch('builtins.print')
    @patch('astropy.io.fits.open')
    def test_dark_info_grab_bad_file(self, mock_fits_open, mock_print):
        mock_fits_open.side_effect = Exception('FITS file error')

        result_df = dark_info_grab(self.filepath, verbose=False)

        expected_data = {
            'name': [self.filename],
            'telescope': ['bad'],
            'exptime': ['bad'],
            'jd': ['bad'],
            'date': ['bad'],
            'filename': [self.filepath]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result_df, expected_df)

        mock_print.assert_called_with('!!! bad ', self.filepath)

    @patch('builtins.print')
    @patch('astropy.io.fits.open')
    def test_dark_info_grab_with_default_verbose(self, mock_fits_open, mock_print):
        mock_fits_header = {
            'TELESCOP': self.telescope_name,
            'EXPTIME': self.exp_time,
            'JD': self.jd,
            'DATE-OBS': self.date
        }
        mock_fits_open.return_value = [MagicMock(header=mock_fits_header)]

        dark_info_grab(self.filepath)

        mock_print.assert_called_with('Done ', self.filepath)


class TestSortDarks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Common mock data for all test cases."""
        cls.fli_dir = 'path/to/fli_dir/'
        cls.existing_files = ['dark1.fit', 'Dark1.fit']
        cls.new_files = ['dark2.fit', 'DARK3.fit']

        cls.dark_list_df = pd.DataFrame(
            {'filename': [cls.fli_dir + cls.existing_files[0], cls.fli_dir + cls.existing_files[1]]}
        )
        cls.mock_new_entries = [pd.DataFrame({'filename': [file]}) for file in cls.new_files]

    @patch('builtins.print')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('pouakai.sort_images.glob')
    @patch('joblib.Parallel')
    def test_sort_darks_no_new_files(self, mock_parallel, mock_glob, mock_to_csv, mock_read_csv, mock_print):
        mock_glob.side_effect = [
            [self.fli_dir + self.existing_files[0]],
            [self.fli_dir + self.existing_files[1]],
            []
        ]
        mock_read_csv.return_value = self.dark_list_df

        sort_darks(verbose=True, num_core=4)

        mock_parallel.assert_not_called()
        mock_to_csv.assert_not_called()
        mock_print.assert_any_call('Number of new darks: ', 0)
        mock_print.assert_any_call('Updated darks')

    @patch('builtins.print')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('pouakai.sort_images.glob')
    @patch('pouakai.sort_images.Parallel')
    def test_sort_darks_with_new_files(self, mock_parallel, mock_glob, mock_to_csv, mock_read_csv, mock_print):
        mock_glob.return_value = [
            [self.fli_dir + self.existing_files[0], self.fli_dir + self.new_files[0]],
            [self.fli_dir + self.existing_files[1], self.fli_dir + self.new_files[1]]
        ]

        mock_read_csv.return_value = self.dark_list_df
        mock_parallel.side_effect = lambda func, *args, **kwargs: lambda _: self.mock_new_entries

        sort_darks(verbose=True, num_core=4)

        mock_parallel.assert_called_once()
        mock_to_csv.assert_called_once()
        mock_print.assert_any_call('Number of new darks: ', len(self.new_files))
        mock_print.assert_any_call('Updated darks')


if __name__ == '__main__':
    unittest.main()
