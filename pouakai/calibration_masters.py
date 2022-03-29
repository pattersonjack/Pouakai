from astropy.io import fits
import pandas as pd
import numpy as np
from glob import glob
import os


def split_names(files):
	names = [x.split('-')[0] for x in files]
	return names


def make_master_darks(save_location = '/home/phys/astronomy/rri38/moa/data/master/dark/',verbose=False):
	# make save_location an environment variable
	dark_list = pd.read_csv('cal_lists/dark_list.csv')
	masters = pd.read_csv('cal_lists/master_dark_list.csv')
	names = split_names(dark_list['name'].values)
	all_names = set(names)
	master_names = set(split_names(masters['name'].values))

	new = all_names ^ master_names
	new = list(new)
	new.sort(reverse=True)
	print('sorted')
	for i in range(len(new)):
		entry = {}
		n = new[i]
		ind = np.array(names) == n
		all_chips = dark_list.iloc[ind]
		print('index 1')

		for j in range(10):
			j += 1
			chip_ind = all_chips['chip'].values == j
			chip = all_chips.iloc[chip_ind]
			chip_files = chip['filename'].values
			print('index 2')
			master = []
			for file in chip_files:
				hdu = fits.open(file)[0]
				header = hdu.header
				data = hdu.data
				master += [data]
			master = np.array(master)
			print('made array')
			if verbose:
				print('Used ',len(master),' images in median')
			m = np.nanmedian(master,axis=0)
			std = np.nanstd(master,axis=0)
			time = np.nanmean(chip['jd'])
			print('calc mean')
			header['JDSTART'] = time 
			header['MASTER'] = True
			phdu = fits.PrimaryHDU(data = m, header = header)
			ehdu = fits.ImageHDU(data = std, header = header)
			hdul = fits.HDUList([phdu, ehdu])


			letter = file.split('-')[2]
			base_name = file.split('/')[-1].split('.')[0].replace(letter,'m')
			save_name = save_location + base_name + '.fits'
			print('saving')
			hdul.writeto(save_name,overwrite=True)
			compress = 'gzip -f ' + save_name
			os.system(compress)
			print('saved')
			entry['name'] = base_name + '.gz'

			entry['chip'] = header['CHIP']
			entry['exptime'] = header['EXPTIME']
			entry['jd'] = time
			entry['date'] = header['DATE-OBS']
			entry['nimages'] = len(master)
			entry['filename'] = save_name + '.gz'
			if len(master) < 3:
				note = 'bad'
			else:
				note = 'good'
			entry['note'] = note
		
			if verbose:
				print('Done ', base_name)
		
			masters = masters.append(entry, ignore_index=True)
			masters.to_csv('cal_lists/master_dark_list.csv',index=False)


def get_master_dark(jd,exptime,chip,strict=True,tol=100):
	"""
	ytdhgvj
	"""
	darks = pd.read_csv('cal_lists/master_dark_list.csv')
	if strict:
		ind = darks['note'].values == 'good'
		darks = darks.iloc[ind]
	dchips = darks['chip'].values
	dexptime = darks['exptime'].values
	chip_ind = dchips == chip

	exp_ind = dexptime == exptime
	ind = chip_ind & exp_ind
	good = darks.iloc[ind]

	if len(good) > 0:
		djd = good['jd'].values
		diff = jd - djd
		min_ind = np.argmin(abs(diff))
		t_diff = diff[min_ind]
		dark = darks.iloc[min_ind]
		fname = dark['filename']
		if abs(t_diff) < tol:
			return fname, t_diff
		else:
			return 'none', -999	
	else:
		return 'none', -999


def cut_bad_reductions(table):
	"""
	remove all reductions with no good dark frames to see if anything better can be done
	"""
	ind = table['time_diff'].values != -999
	return table.iloc[ind]


def make_master_flats(save_location = '/home/phys/astronomy/rri38/moa/data/master/flat/',redo_bad=True, verbose=False):
	# make save_location an environment variable
	flat_list = pd.read_csv('cal_lists/flat_list.csv')
	masters = pd.read_csv('cal_lists/master_flat_list.csv')
	if redo_bad:
		masters = cut_bad_reductions(masters)
	names = split_names(flat_list['name'].values)
	all_names = set(names)
	master_names = set(split_names(masters['name'].values))
	print(all_names)
	new = all_names ^ master_names
	new = list(new)
	print(new)
	new.sort(reverse=True)
	for i in range(len(new)):
		entry = {}
		n = new[i]
		ind = np.array(names) == n
		all_chips = flat_list.iloc[ind]
		dark_get = True
		for j in range(10):
			j += 1
			chip_ind = all_chips['chip'].values == j
			chip = all_chips.iloc[chip_ind]
			chip_files = chip['filename'].values
			master = []
			for file in chip_files:
				hdu = fits.open(file)[0]
				header = hdu.header
				data = hdu.data.astype(float)

				saturations = (data > 40000).flatten()
				# if more than 10% of pixels are saturated, set array to nan
				if sum(saturations) > len(saturations)*0.1:
					data = data * np.nan

				master += [data]
			master = np.array(master)
			if verbose:
				print('Used ',len(master),' images in median')
			# get dark frame
			if dark_get:
				fname, tdiff = get_master_dark(chip['jd'].values[0], chip['exptime'].values[0], j)
				dark_name = fname.split(str(j)+'.fits.gz')[0]
				d_tdiff = tdiff
				dark_get = False
			else:
				fname = dark_name + str(j) + '.fits.gz'
				tdiff = d_tdiff
			if verbose:
				print('using dark frame ',fname)
				print('time difference ',tdiff)
			try:
				dark = fits.open(fname)[0].data
				master = master - dark
			except:
				m = '!!! Warning: No dark found !!!'
				print(m)
				tdiff = -999
			
			m = np.nanmedian(master,axis=0)
			std = np.nanstd(master,axis=0)
			time = np.nanmean(chip['jd'])
			header['JDSTART'] = time 
			header['MASTER'] = True
			phdu = fits.PrimaryHDU(data = m, header = header)
			ehdu = fits.ImageHDU(data = std, header = header)
			hdul = fits.HDUList([phdu, ehdu])


			letter = file.split('-')[3]
			base_name = file.split('/')[-1].split('.')[0].replace(letter,'m')
			save_name = save_location + base_name + '.fits'
			print('saving')
			hdul.writeto(save_name,overwrite=True)
			compress = 'gzip -f ' + save_name
			os.system(compress)
			print('saved')
			entry['name'] = base_name + '.gz'

			entry['band'] = header['COLOUR']
			entry['chip'] = header['CHIP']
			entry['exptime'] = header['EXPTIME']
			entry['jd'] = time
			entry['date'] = header['DATE-OBS']
			entry['filename'] = save_name
			entry['dark_file'] = fname 
			entry['time_diff'] = tdiff
			entry['nimages'] = len(master)
			if (np.nanmedian(m) < 15000):
				note = 'bad'
			else:
				note = 'good'
			entry['note'] = note
			
			field = header['FIELD']
			if 'flat_round' in field:
				flat_type = 'dome'
			else:
				flat_type = 'sky'
			entry['field'] = field
			entry['flat_type'] = flat_type

			if verbose:
				print('Done ', base_name)
		
			masters = masters.append(entry, ignore_index=True)
			masters.to_csv('cal_lists/master_flat_list.csv',index=False)


if __name__=='__main__':
	make_master_darks(verbose=True)
	print('!!! Finished darks !!!')
	make_master_flats(verbose=True)
	print('!!! Finished flats !!!')
