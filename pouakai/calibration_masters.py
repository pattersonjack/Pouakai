from astropy.io import fits
import pandas as pd
import numpy as np
from glob import glob
import os
from copy import deepcopy
from joblib import Parallel, delayed

def split_names(files):
	names = [x.split('-')[0] for x in files]
	return names


def make_master_darks(save_location = '/home/phys/astronomy/rri38/moa/data/master/dark/',num_cores=25,verbose=False):
	# make save_location an environment variable
	dark_list = pd.read_csv('cal_lists/dark_list.csv')
	ind = dark_list['chip'].values != 'bad'
	dark_list = dark_list[ind]
	masters = pd.read_csv('cal_lists/master_dark_list.csv')
	names = split_names(dark_list['name'].values)
	all_names = set(names)
	master_names = set(split_names(masters['name'].values))

	new = all_names - master_names
	new = list(new)
	new.sort(reverse=True)
	print('sorted')
	indexer = np.arange(len(new),dtype=int)
	if num_cores > 1:
		entries = Parallel(n_jobs=num_cores)(delayed(dark_processing)(index,new,names,dark_list,save_location,verbose) for index in indexer)
	else:
		entries = []
		for index in indexer:
			entries += [dark_processing(index,new,names,dark_list,save_location,verbose)]
			print('!!! ', entries)
	for entry in entries:
		masters = masters.append(entry,ignore_index=True)
	masters.to_csv('cal_lists/master_dark_list.csv',index=False)

def dark_processing(index,new,names,dark_list,save_location,verbose):
	entries = {}
	n = new[index]
	ind = np.array(names) == n
	all_chips = dark_list.iloc[ind]
	#print(all_chips)
	for j in range(10):
		try:
			j += 1
			entry = {}
			chip_ind = all_chips['chip'].values.astype(int) == j
			chip = all_chips.iloc[chip_ind]
			chip_files = chip['filename'].values
			master = []
			for file in chip_files:
				hdu = fits.open(file)[0]
				header = hdu.header
				data = hdu.data
				master += [data]
			master = np.array(master)
			#print('made array')
			if verbose:
				print('Used ',len(master),' images in median')
			m = np.nanmedian(master,axis=0)
			std = np.nanstd(master,axis=0)
			time = np.nanmean(chip['jd'].astype(float))
			#print('calc mean')
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
			entry['name'] = base_name

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
			if len(entries) == 0:
				entries = pd.DataFrame([entry])
			else:
				entries = entries.append(entry,ignore_index=True)
		except:
			print('bad')
		
	return entries 

def get_master_dark(jd,exptime,chip,strict=True,tol=1):
	"""
	ytdhgvj
	"""
	darks = pd.read_csv('cal_lists/master_dark_list.csv')
	if strict:
		ind = darks['note'].values == 'good'
		darks = darks.iloc[ind]
	dchips = darks['chip'].values
	chip_ind = dchips == chip
	darks = darks.iloc[chip_ind]
	
	dexptime = darks['exptime'].values
	exp_ind = dexptime.astype(int) == int(exptime)
	good = darks.iloc[exp_ind]

	if len(good) > 0:
		djd = good['jd'].values
		diff = jd - djd
		min_ind = np.argmin(abs(diff))
		t_diff = diff[min_ind]
		dark = good.iloc[min_ind]
		fname = dark['filename']
		#print('flat exp:{}, chip:{}'.format(exptime,chip))
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
	ind = table['time_diff'].values == -999
	bad_names = set(split_names(table['name'].iloc[ind]))
	names = split_names(table['name'])
	bad_names = list(bad_names)
	tab = deepcopy(table)
	for i in range(len(bad_names)):
		print('Dropping ' + bad_names[i])
		inds = np.where(names != bad_names[i])[0]
		tab = tab.iloc[inds]
		names = split_names(tab['name'])
	return tab


def make_master_flats(save_location = '/home/phys/astronomy/rri38/moa/data/master/flat/',redo_bad=False, verbose=False):
	# make save_location an environment variable
	flat_list = pd.read_csv('cal_lists/flat_list.csv')
	masters = pd.read_csv('cal_lists/master_flat_list.csv')
	if redo_bad:
		masters = cut_bad_reductions(masters)
	names = split_names(flat_list['name'].values)
	all_names = set(names)
	master_names = set(split_names(masters['name'].values))
	print(len(all_names))
	new = all_names - master_names
	new = list(new)
	print(new)
	print(len(new))
	new.sort(reverse=True)
	for i in range(len(new)):
		print(str(i) + ' of ' + str(len(new)))
		entry = {}
		n = new[i]
		ind = np.array(names) == n
		all_chips = flat_list.iloc[ind]

		dark_get = True
		for j in range(10):
			j += 1
			chip_ind = all_chips['chip'].values.astype(int) == j
			chip = all_chips.iloc[chip_ind]
			chip_files = chip['filename'].values
			master = []
			for file in chip_files:
				hdu = fits.open(file)[0]
				header = hdu.header
				data = hdu.data.astype(float)

				saturations = (data > 50000).flatten()
				# if more than 10% of pixels are saturated, set array to nan
				if sum(saturations) > len(saturations) * 0.1:
					print('image ', file, ' is saturated')
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
				if dark_name != 'none':
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
			
			mas = np.nanmedian(master,axis=0)
			std = np.nanstd(master,axis=0)
			time = np.nanmean(chip['jd'])
			header['JDSTART'] = time 
			header['MASTER'] = True
			phdu = fits.PrimaryHDU(data = mas, header = header)
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
			entry['name'] = base_name

			entry['band'] = header['COLOUR'].strip()
			entry['chip'] = header['CHIP']
			entry['exptime'] = header['EXPTIME']
			entry['jd'] = time
			entry['date'] = header['DATE-OBS']
			entry['filename'] = save_name + '.gz'
			entry['dark_file'] = fname 
			entry['time_diff'] = tdiff
			entry['nimages'] = len(master)

			field = header['FIELD']
			if 'flat_round' in field:
				flat_type = 'dome'
			else:
				flat_type = 'sky'
			entry['field'] = field
			entry['flat_type'] = flat_type

			if (np.nanmedian(mas) < 15000) | (np.nansum(mas) <= 0):
				note = 'bad'
			else:
				if (len(master) < 2) & (flat_type == 'dome'):
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

def new_make_master_flats(save_location = '/home/phys/astronomy/rri38/moa/data/master/flat/',time_frame=60,num_cores=25, verbose=False):
	# make save_location an environment variable
	flat_list = pd.read_csv('cal_lists/flat_list.csv')
	masters = pd.read_csv('cal_lists/master_flat_list.csv')
	flat_list['field'] = flat_list['field'].str.strip()
	ind = (flat_list['field'].values == 'flat_round') & (flat_list['note'].values == 'good') & (flat_list['chip'].values != 99) & (flat_list['chip'].values != 0)
	flat_list = flat_list.iloc[ind]
	flat_list['band'] = flat_list['band'].str.strip()
	times = flat_list['jd'].values.astype(int)
	names = []
	for i in range(len(times)):
		names += [('F' + times.astype(str)[i] + '_' + str(time_frame) + 'd_' + flat_list['band'].values[i]
			 		+ '_' + flat_list['chip'].values.astype(str)[i])]

	all_names = set(names)
	master_names = set(split_names(masters['name'].values))
	new = all_names - master_names
	new = list(new)
	print('Number of new flat entries: ',len(new))
	if len(new) > 0:
		new.sort(reverse=True)
		indexer = np.arange(len(new))
		entries = Parallel(n_jobs=num_cores)(delayed(flat_processing)(index,new,flat_list,times,time_frame,save_location,verbose) for index in indexer)
		entries = pd.concat(entries,ignore_index=True)	
		
		masters = masters.append(entries, ignore_index=True)
		masters.to_csv('cal_lists/master_flat_list.csv',index=False)

def flat_processing(index,new,flat_list,times,time_frame,save_location,verbose):
	i = index
	entry = {}
	n = new[i]
	t = int(n[1:].split('_')[0])
	c = int(n.split('_')[-1])
	b = n.split('_')[2]

	tind = (t - times >= 0) & (t - times <= time_frame)
	cind = flat_list['chip'].values == c
	bind = flat_list['band'].values == b
	ind = tind & cind & bind

	files = flat_list['filename'].values[ind]
	exptimes = flat_list['exptime'].values[ind]
	if verbose:
		print('Num flats: ', len(files))

	master_arr = []
	darks = []
	if len(files) > 10:
		files = files[:10]
	for j in range(len(files)):
		
		hdu = fits.open(files[j])[0]
		header = hdu.header
		#print(header)
		data = hdu.data.astype(float)

		saturations = (data > 50000).flatten()
		# if more than 10% of pixels are saturated, set array to nan
		if sum(saturations) > len(saturations) * 0.1:
			print('image ', files[j], ' is saturated')
			data = data * np.nan
		master_arr += [data]

		fname, tdiff = get_master_dark(t, exptimes[j], c)
		try:
			darks += [fits.open(fname)[0].data]
		except:
			darks += [data * np.nan]

	master_arr = np.array(master_arr)
	darks = np.array(darks)
	master_arr = master_arr - darks

	mas = np.nanmean(master_arr,axis=0)
	std = np.nanstd(master_arr,axis=0)
	header['JDSTART'] = t 
	header['MASTER'] = True
	phdu = fits.PrimaryHDU(data = mas, header = header)
	ehdu = fits.ImageHDU(data = std, header = header)
	hdul = fits.HDUList([phdu, ehdu])

	save_name = save_location + n + '.fits'
	print('saving')
	hdul.writeto(save_name,overwrite=True)
	compress = 'gzip -f ' + save_name
	os.system(compress)
	print('saved')
	entry['name'] = n

	entry['band'] = header['COLOUR'].strip()
	entry['chip'] = header['CHIP']
	entry['exptime'] = header['EXPTIME']
	entry['jd'] = t
	entry['date'] = header['DATE-OBS']
	entry['filename'] = save_name + '.gz'
	entry['nimages'] = len(master_arr)

	field = header['FIELD']
	if 'flat_round' in field:
		flat_type = 'dome'
	else:
		flat_type = 'sky'
	entry['field'] = field
	entry['flat_type'] = flat_type

	if (np.nanmedian(mas) < 15000) | (np.nansum(mas) <= 0):
		note = 'bad'
	else:
		if (len(master_arr) < 2) & (flat_type == 'dome'):
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
		print('Done ', n)
	return pd.DataFrame([entry])
	

def make_masters(verbose=True):
	make_master_darks(verbose=True)
	if verbose:
		print('!!! Finished darks !!!')
	new_make_master_flats(verbose=True)
	if verbose:
		print('!!! Finished flats !!!')

if __name__ == '__main__':
	make_masters(verbose=True)