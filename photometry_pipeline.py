import sys, subprocess, optparse, os, glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#from pymongo import MongoClient

from skimage import data, img_as_float
from skimage import exposure

from sklearn import preprocessing

from astropy.io import fits

# for source detection
from photutils import daofind
from astropy.stats import median_absolute_deviation as mad
from astropy.stats import mad_std

# for SDSS cross match
from astropy.wcs import WCS
from astroquery.sdss import SDSS
from astropy import units as u
from astropy import coordinates as coords

# for psf photometry
from pyraf import iraf

sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/")
from convolutional_sparseFiltering import convolve, pool, get_sparseFilter

sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/tools/")
from classify import predict

np.seterr(all="ignore")

def convolve_and_pool(X, imageDim=20, patchDim=6, poolDim=5, numFeatures=400, stepSize=40):
    
    patchesFile = "/Users/dew/development/PS1-Real-Bogus/data/patches_stl-10_unlabeled_meansub_20150409_psdb_6x6.mat"
    patches = sio.loadmat(patchesFile)["patches"].T
    SF = get_sparseFilter(400, patches, patchesFile, maxiter=100)
    W = np.reshape(SF.trainedW, (SF.k, SF.n), order="F")
    SF = None
    patches = None
    numTrainImages = np.shape(X)[0]
    trainImages = np.zeros((imageDim,imageDim,1,numTrainImages))
    for i in range(numTrainImages):
        image = np.reshape(X[i,:], (imageDim,imageDim), order="F")
        trainImages[:,:,0,i] = trainImages[:,:,0,i] + image
    X = None
    pooledFeaturesTrain = np.zeros((numFeatures,numTrainImages, \
                                    int(np.floor((imageDim-patchDim+1)/poolDim)), \
                                    int(np.floor((imageDim-patchDim+1)/poolDim))))

    for convPart in range(numFeatures/stepSize):
        featureStart = convPart*stepSize
        featureEnd = (convPart+1)*stepSize
        print '[*] Step %d: features %d to %d'% (convPart, featureStart, featureEnd)
        Wt = W[featureStart:featureEnd, :]
        print '   [*] Convolving and pooling images'
        convolvedFeaturesThis = convolve(patchDim, stepSize, trainImages, Wt)
        pooledFeaturesThis = pool(poolDim, convolvedFeaturesThis)
        pooledFeaturesTrain[featureStart:featureEnd, :, :, :] += pooledFeaturesThis
        convolvedFeaturesThis = pooledFeaturesThis = None
    return pooledFeaturesTrain

def signPreserveNorm(Vec):
    std = np.std(Vec)
    normVec = ((Vec)/ np.abs(Vec))*(np.log1p(np.abs(Vec)/std))
    return normVec

def prepare_data_for_ML(data, sources):
    
    m = len(sources['xcentroid'])
    X = np.zeros((m,400))
    #dim = np.ceil(np.sqrt(m))
    #fig = plt.figure()
    for i in range(m):
    #    ax = fig.add_subplot(dim,dim,i+1)
        position = (sources['xcentroid'][i],sources['ycentroid'][i])
        #print i,position[1], position[0]
        img = data[position[1]-10:position[1]+10,position[0]-10:position[0]+10]
    #    ax.imshow(img, cmap="gray_r", interpolation="nearest")
    #    plt.axis("off")
        try:
            assert np.shape(np.ravel(img, order="F"))[0] == 400
            X[i,:] += np.nan_to_num(signPreserveNorm(np.nan_to_num(np.ravel(img, order="F"))))
        except AssertionError:
            continue
    #plt.show()
    return X

def apply_ML(data, sources):
    
    decsion_boundary = 0.159
    
    # prepare data for machine learning
    print "[*] Preparing data for machine learning."
    X = prepare_data_for_ML(data, sources)

    pooledFeatures = convolve_and_pool(X)

    print "[*] Applying feature scaling."
    tmp = sio.loadmat("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/"+\
                      "SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_"+\
                      "stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat")["pooledFeaturesTrain"]
    tmp = np.transpose(tmp, (0,2,3,1))
    numTrainImages = np.shape(tmp)[3]
    tmp = np.reshape(tmp, (int((tmp.size)/float(numTrainImages)), \
                           numTrainImages), order="F")
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(tmp.T)  # Don't cheat - fit only on training data
    tmp = None

    X = np.transpose(pooledFeatures, (0,2,3,1))
    numImages = np.shape(X)[3]
    X = np.reshape(X, (int((pooledFeatures.size)/float(numImages)), \
                       numImages), order="F")
    X = scaler.transform(X.T)
    clfFile = "/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/classifiers/"+\
              "SVM_linear_C1.000000e+00_SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm"+\
              "_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.pkl"
    print "[*] Making predictions."
    pred = predict(clfFile, X)
    m = len(np.where(pred > decsion_boundary)[0])
    print "[*] %d quality detections passing machine learning threshold (5%% FoM)." % (m)
    """
    print "[*] Plotting quality detections."
    dim = np.ceil(np.sqrt(m))
    fig = plt.figure()
    for i, index in  enumerate(np.where(pred > decsion_boundary)[0]):
        ax = fig.add_subplot(dim,dim,i+1)
        position = (sources['xcentroid'][index],sources['ycentroid'][index])
        img = data[position[1]-10:position[1]+10,position[0]-10:position[0]+10]
        ax.imshow(img, cmap="gray_r", interpolation="nearest", origin="lower")
        plt.axis("off")
    plt.show()
    """
    return np.where(pred > decsion_boundary)[0]

def load_image_data(imageFile, extent, extension=0):
    # open the image fits file
    print "[*] Opening %s" % (imageFile)
    hdulist = fits.open(imageFile)
    # the pixel dimensions of image
    size = np.shape(hdulist[extension].data)
    # extract seeing from header
    #print "[*] Extracting seeing."
    #seeing_in_pix = hdulist[1].header["CHIP.SEEING"]
    #effective_gain = hdulist[1].header["HIERARCH CELL.GAIN"]
    #filter = hdulist[1].header["HIERARCH FPA.FILTERID"].split(".")[0]
    # calculate centre of image
    print "[*] Calculating mid-point of image."
    mid_point = np.shape(np.nan_to_num(hdulist[extension].data))[0] / 2.0
    print "[*] Extracting (%d,%d) substamp from image at mid-point." % (extent, extent)
    # extract the pixel data on which to perform photometry
    data = np.nan_to_num(hdulist[extension].data)[mid_point-extent:mid_point+extent,\
                                          mid_point-extent:mid_point+extent]
    # extract image to plot for visual check
    image = img_as_float(np.nan_to_num(hdulist[extension].data) / np.max(np.nan_to_num(hdulist[extension].data)))
    # histogram equalize image to ensure image scale is easy to visualise
    image = exposure.equalize_hist(image)[mid_point-extent:mid_point+extent,\
                                          mid_point-extent:mid_point+extent]
    return data, image, hdulist, size, (mid_point, mid_point)

def find_sources(imageFile, data, seeing_in_pix, threshold=5.):
    # estimate the 1-sigma noise level using the median absolute deviation of the image
    print "[*] Estimating 1-sigma noise level."
    # generate a mask for 0 pixel counts. These are chip gaps or skycell edges generated by
    # np.nan_to_num and will affect noise level estimate.
    mask = np.where(data != 0)
    bkg_sigma = mad_std(data[mask])
    #print np.median(data), mad(data), bkg_sigma
    # use daofind to detect sources setting
    print "[*] Detecting %d-sigma sources in %s" % (threshold, imageFile)
    sources = daofind(data, fwhm=seeing_in_pix, threshold=threshold*bkg_sigma)
    print "[*] Source detection successful."
    print "\t[i] %d sources detected: " % (len(sources["xcentroid"]))
    print
    print sources
    return sources, bkg_sigma

def extract_dao_params(dao_param_file):
    input = open(dao_param_file,"r")
    
    sky_mean = []
    sky_sigma = []
    FWHMpsf = []
    for line in input.readlines():
        if "#" in line or line == "\n":
            continue
        data = line.rstrip().split()
        print data
        sky_mean.append(float(data[2]))
        sky_sigma.append(float(data[3]))
        FWHMpsf.append(float(data[4]))
    print
    print "[*] Mean SKY : %f" % np.mean(sky_mean)
    print "[*] Mean SKYSIGMA : %f" % np.mean(sky_sigma)
    print "[*] Mean FWHM : %f" % np.mean(FWHMpsf)
    print "[*] Estimated datamin : %f" % (np.mean(sky_mean) - (5*np.mean(sky_sigma)))
    print
    aperture_r = 1.5*np.mean(FWHMpsf)
    print "[*] Aperture radius (1.5*mean_fwhm) : %f" % (aperture_r)
    print "[*] Inner annulus radius (4 * aperture radius) : %f" % (4*aperture_r)
    print "[*] Outer annulus radius (dannulus) (6 * aperture radius) : %f" % (6*aperture_r)
    print
    return np.mean(sky_mean),np.mean(sky_sigma),np.mean(FWHMpsf),(np.mean(sky_mean) - (5*np.mean(sky_sigma))), aperture_r,4*aperture_r,6*aperture_r

def extract_header_info(hdulist, extension=0):
    
    gain_key      = "HIERARCH CELL.GAIN"
    readnoise_key = "HIERARCH CELL.READNOISE"
    airmass_key   = "AIRMASS"
    exptime_key   = "EXPTIME"
    obstime_key   = "HIERARCH FPA.SHUTOUTC"
    datamax_key   = "HIERARCH CELL.SATURATION"
    filter_key    = "HIERARCH FPA.FILTERID"

    datamax   = float(hdulist[extension].header[datamax_key])
    ccdread   = readnoise_key
    gain      = gain_key
    readnoise = float(hdulist[extension].header[readnoise_key])
    epadu     = float(hdulist[extension].header[gain_key])
    exposure  = exptime_key
    airmass   = airmass_key
    filter    = filter_key
    obstime   = obstime_key
    itime     = float(hdulist[extension].header[exptime_key])
    xairmass  = float(hdulist[extension].header[airmass_key])
    ifilter   = hdulist[extension].header[filter_key]
    otime     = hdulist[extension].header[obstime_key]
    print
    print "[*] (datamax)   : %f" % datamax
    print "[*] (ccdread)   : %s" % ccdread
    print "[*] (gain)      : %s" % gain
    print "[*] (readnoise) : %f" % readnoise
    print "[*] (epadu)     : %f" % epadu
    print "[*] (exposure)  : %s" % exposure
    print "[*] (airmass)   : %s" % airmass
    print "[*] (filter)    : %s" % filter
    print "[*] (obstime)   : %s" % obstime
    print "[*] (itime)     : %f" % itime
    print "[*] (xairmass)  : %f" % xairmass
    print "[*] (ifilter)   : %s" % ifilter
    print "[*] (otime)     : %s" % otime
    print

    return datamax, ccdread, gain, readnoise, epadu, exposure, airmass, filter, \
           obstime, itime, xairmass, ifilter, otime


def transform_to_ps1_bandpass(filter, mag_sdss, mag_err_sdss, gr_sdss, gr_sdss_err):
    # transform coefficients from Tonry et al. 2012 p.g. 25
    transform_coeffs = {"g":{"B0":-0.012, "B1":-0.139, "err":0.007},
                        "r":{"B0": 0.000, "B1":-0.007, "err":0.002},
                        "i":{"B0": 0.004, "B1":-0.014, "err":0.003},
                        "z":{"B0":-0.013, "B1": 0.039, "err":0.009},
                        "y":{"B0": 0.031, "B1":-0.095, "err":0.024},
                        "w":{"B0": 0.012, "B1": 0.039, "err":0.025}}

    ps1_mag = transform_coeffs[filter]["B0"] + gr_sdss*transform_coeffs[filter]["B1"] + mag_sdss
    ps1_mag_err = np.sqrt(transform_coeffs[filter]["err"]*transform_coeffs[filter]["err"] + mag_err_sdss*mag_err_sdss + gr_sdss_err*gr_sdss_err)
    return ps1_mag, ps1_mag_err

def check_legacy_target1_flags(id):
    stellar_valid_flags = [13,14,15,16,18,19]
    query = "'select legacy_target1 from SpecPhotoAll where objID = %s'" % id
    cmd = "python sqlcl_dr9.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        #print result
        if result[0] == "No objects have been found":
            return True
        return int(result[1]) & (2**stellar_valid_flags[0] | 2**stellar_valid_flags[1] | 2**stellar_valid_flags[2] | 2**stellar_valid_flags[3] | 2**stellar_valid_flags[4] | 2**stellar_valid_flags[5]) > 0
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        return True
    except IndexError:
        return True

def check_boss_target1_flags(id):
    stellar_valid_flags = [20,21,34,35]
    query = "'select boss_target1 from SpecPhotoAll where objID = %s'" % id
    cmd = "python sqlcl_dr9.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        #print result
        if result[0] == "No objects have been found":
            return True
        return int(result[1]) & (2**stellar_valid_flags[0] | 2**stellar_valid_flags[1] | 2**stellar_valid_flags[2] | 2**stellar_valid_flags[3] | 2**stellar_valid_flags[4]) > 0
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        return True
    except IndexError:
        return True

def get_reference_mags(id, filter):
    # setup mongo database connection
    #client = MongoClient('localhost:27017')
    #db = client.ps1gw_reference_star_db
    
    if filter == "y":
        query_filter = "z"
    elif filter == "w":
        query_filter = "r"
    else:
        query_filter = filter
    #try:
    #    query = {"_id":{"$eq":id}}
    #    results = db.ps1gw_reference_star_db.find(query)
    #except pymongo.errors.DuplicateKeyError:
    #    print "[!] No mongoDB entry for %s" % str(id)
    query = "'select %s, err_%s, g, err_g, r, err_r, type from PhotoObjAll where objID = %s'" % \
                (query_filter, query_filter, id)
    cmd = "python sqlcl.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        # ensure object is a SDSS star with r < 21
        if result[0] == "No objects have been found":
            print "[!] No objects have been found."
            return "NULL", "NULL"
        elif result[7] != "6":
            print "[!] SDSS reports as galaxy."
            return "NULL", "NULL"
        elif float(result[5]) > 21.0:
            print "[!] SDSS r mag > 21, unreliable SDSS star-galaxy separation."
            return "NULL", "NULL"
        elif float(result[5]) < 15.0:
            print "[!] SDSS r mag < 15, PS1 detection may be saturated."
            return "NULL", "NULL"
        mag_sdss = float(result[1])
        mag_err_sdss = float(result[2])
        gr_sdss = float(result[3]) - float(result[5])
        gr_sdss_err = np.sqrt(float(result[4])*float(result[4])+float(result[6])*float(result[6]))
        # ensure oject has no QSO flags set
        if check_legacy_target1_flags(id) and check_boss_target1_flags(id):
            return transform_to_ps1_bandpass(filter, mag_sdss, mag_err_sdss, gr_sdss, gr_sdss_err)
        else:
            # else reject object
            print "[!] failed QSO check."
            return "NULL", "NULL"
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        print "[!] subprocess exit error."
        return "NULL", "NULL"
    except IndexError:
        print "[!] index error."
        return "NULL", "NULL"

def house_keeping(path, imageFile, diffFile):
    
    try:
        assert os.path.isfile(imageFile.strip(".fits")+"_dao_params.txt")
    except AssertionError:
        print "[!] %s must exist before running this program.  From iraf/pyraf and run daoedit on 5 stars in %s" % (imageFile.strip(".fits")+"_dao_params.txt", imageFile)
        raise AssertionError

    try:
        for file in glob.glob(imageFile+".*"):
            os.remove(file)
    except OSError:
        pass
    try:
        for file in glob.glob(diffFile+".*"):
            os.remove(file)
    except OSError:
        pass
    return None

def get_detection_positions(sources, extent, mid_point):
    positions = []
    # check transient is detected
    transient_position = None
    transient_index = None
    for id in sources["id"]:
        # if detected record position
        if np.isclose(sources['xcentroid'][id-1], extent, atol=2) and np.isclose(sources['ycentroid'][id-1], extent, atol=2):
            transient_position = (sources['xcentroid'][id-1], sources['ycentroid'][id-1])
            transient_index = id - 1
        else:
            positions.append((sources['xcentroid'][id-1], sources['ycentroid'][id-1]))
    # else set position to central pixel (valid assumption for PS1 postage stamps)
    if transient_index == None:
        transient_position = mid_point
    try:
        print "[*] Transient index = %d " % transient_index
        # and remove from detected sources
        sources.remove_row(transient_index)
    except TypeError:
        print "[!] daofind() did not detect transient!"
        print "[+] Transient position set to image mid-point."
    return sources, positions, transient_position, transient_index

def cross_match_sdss(imageFile, extension, sources, quality_detections, extent, search_radius=3):
    
    
    data, image, hdulist, size, mid_point = load_image_data(imageFile, extent, extension=extension)
    # garbage collect data and image
    data = None
    image = None
    
    # cross match quality detections with SDSS
    filter = hdulist[extension].header["HIERARCH FPA.FILTERID"].split(".")[0]
    
    # get WCS info from fits header
    wcs = WCS(hdulist[extension].header)
    
    reference_dict = {}
    pos_output = open(imageFile+".quality.coo", "w")
    count = 0
    for index in quality_detections:
        if count == 30:
            break
        print
        pixcrd = np.array([[sources['xcentroid'][index]+((size[0]/2.0)-extent), \
                            sources['ycentroid'][index]+((size[1]/2.0)-extent)]], np.float_)
            
        world = wcs.wcs_pix2world(pixcrd, 1)
        pos = coords.SkyCoord(ra=world[0][0]*u.degree,dec=world[0][1]*u.degree, frame='icrs')
        # search ? arsec region in SDSS
        xid = SDSS.query_region(pos, radius=search_radius*(1/3600.0)*u.degree)
        print xid
        try:
            for i in range(len(xid["objid"])):
                id = xid["objid"][i]
                mag, mag_err = get_reference_mags(id, filter)
                if mag == "NULL":
                    continue
                break
        except TypeError:
            continue
        print
        print "[*] %s, %s, %s, %s" % (id, str((sources['xcentroid'][index], sources['ycentroid'][index])), str(mag), str(mag_err))
        if mag != "NULL" and mag_err != "NULL":
            reference_dict[index] = {"mag":float(mag), "mag_err":float(mag_err), \
                                     "id":id, "pos":(sources['xcentroid'][index], \
                                                     sources['ycentroid'][index])}
            pos_output.write("%s %s\n" % (str(pixcrd[0][0]), str(pixcrd[0][1])))
        count += 1
    pos_output.close()
    return reference_dict

def doaphot_psf_photometry(path, imageFile, extent, extension):
    
    data, image, hdulist, size, mid_point = load_image_data(imageFile, extent, extension=extension)
    # garbage collect data and image
    data = None
    image = None
    
    # import IRAF packages
    iraf.digiphot(_doprint=0)
    iraf.daophot(_doprint=0)

    # dao_params.txt must be created manually using daoedit in iraf/pyraf for 5 stars
    dao_params = extract_dao_params(imageFile.strip(".fits")+"_dao_params.txt")
    
    sky                  = dao_params[0]
    sky_sigma            = dao_params[1]
    fwhm                 = dao_params[2]
    datamin              = dao_params[3]
    aperature_radius     = dao_params[4]
    annulus_inner_radius = dao_params[5]
    annulus_outer_radius = dao_params[6]
    
    # get datapars
    datapars = extract_header_info(hdulist)
    
    datamax   = datapars[0]
    ccdread   = datapars[1]
    gain      = datapars[2]
    readnoise = datapars[3]
    epadu     = datapars[4]
    exposure  = datapars[5]
    airmass   = datapars[6]
    filter    = datapars[7]
    obstime   = datapars[8]
    itime     = datapars[9]
    xairmass  = datapars[10]
    ifilter   = datapars[11]
    otime     = datapars[12]
    
    # set datapars
    iraf.datapars.unlearn()
    iraf.datapars.setParam('fwhmpsf',fwhm)
    iraf.datapars.setParam('sigma',sky_sigma)
    iraf.datapars.setParam('datamin',datamin)
    iraf.datapars.setParam('datamax',datamax)
    iraf.datapars.setParam('ccdread',ccdread)
    iraf.datapars.setParam('gain',gain)
    iraf.datapars.setParam('readnoise',readnoise)
    iraf.datapars.setParam('epadu',epadu)
    iraf.datapars.setParam('exposure',exposure)
    iraf.datapars.setParam('airmass',airmass)
    iraf.datapars.setParam('filter',filter)
    iraf.datapars.setParam('obstime',obstime)
    iraf.datapars.setParam('itime',itime)
    iraf.datapars.setParam('xairmass',xairmass)
    iraf.datapars.setParam('ifilter',ifilter)
    iraf.datapars.setParam('otime',otime)
    
    # set photpars
    iraf.photpars.unlearn()
    iraf.photpars.setParam('apertures',aperature_radius)
    zp_estimate = iraf.photpars.getParam('zmag')
    # set centerpars
    iraf.centerpars.unlearn()
    iraf.centerpars.setParam('calgorithm','centroid')
    iraf.centerpars.setParam('cbox',5.)
    
    # set fitskypars
    iraf.fitskypars.unlearn()
    iraf.fitskypars.setParam('annulus',annulus_inner_radius)
    iraf.fitskypars.setParam('dannulus',annulus_outer_radius)
    
    # run phot
    run_phot(imageFile, imageFile+".quality.coo")
    
    # set daopars
    iraf.daopars.unlearn()
    iraf.daopars.setParam('function','auto')
    iraf.daopars.setParam('psfrad', 2*int(fwhm)+1)
    iraf.daopars.setParam('fitrad', fwhm)
    
    # select a psf/prf star
    # taking whatever the default selection is, can't see a way to pass coords of desired
    # stars, if could would use those in dao_params.txt
    # An alternative is to reorder the objects so those in dao_params.txt are at top of
    # sources table, assuming those are the defaults selected here.
    iraf.pstselect.unlearn()
    iraf.pstselect.setParam('maxnpsf',5)
    iraf.pstselect(image=imageFile,photfile=imageFile+".mags.1",pstfile=imageFile+".pst.1",interactive='no')
    
    # fit the psf
    iraf.psf.unlearn()
    iraf.psf(image=imageFile, \
             photfile=imageFile+".mags.1",\
             pstfile=imageFile+".pst.1",\
             psfimage=imageFile+".psf.1.fits",\
             opstfile=imageFile+".pst.2",\
             groupfile=imageFile+".psg.1",\
             interactive='no')
    # check the psf
    # perhaps pass it through ML and visualise
    # save visualisation for later manual checks
    iraf.seepsf.unlearn()
    iraf.seepsf(psfimage=imageFile+".psf.1.fits", image=imageFile+".psf.1s.fits")
             
    hdulist_psf = fits.open(imageFile+".psf.1s.fits")
    #print "[*] plotting PSF for visual check."
    #plt.imshow(hdulist_psf[0].data, interpolation="nearest",cmap="hot")
    #plt.axis("off")
    #plt.show()
             
    # perform photometry
    run_allstar(imageFile,imageFile+".psf.1.fits")
    return zp_estimate, imageFile+".psf.1.fits"

def run_phot(imageFile, coords):
    iraf.phot.unlearn()
    iraf.phot(image=imageFile,coords=coords,output=imageFile+".mags.1",interactive='no')

def run_allstar(imageFile, psfimage):
    iraf.allstar.unlearn()
    iraf.allstar(image=imageFile,\
                 photfile=imageFile+".mags.1",\
                 psfimage=psfimage, \
                 allstarfile=imageFile+".als.1",\
                 rejfile=imageFile+".arj.1",\
                 subimage=imageFile+".sub.1")

def calculate_zeropoint_offset(reference_dict, measurement_dict, num_check):
    
    diffs = []
    #num_measurements = np.max(measurement_dict.keys())
    num_measurements = len(measurement_dict.keys())
    # using iraf keys in reference and measurement dicts no longer work
    # need to generate a mapping.
    # .als.1 file is ordered by y pixel coordinate.
    
    rkeys = reference_dict.keys()
    y_pos = []
    for key in rkeys:
        y_pos.append(reference_dict[key]["pos"][1])

    sorted_rkeys = [list(x) for x in zip(*sorted(zip(rkeys, y_pos), key=lambda pair: pair[1]))][0]

    for key in measurement_dict.keys():
        if key < num_measurements - num_check:
            diff = reference_dict[sorted_rkeys[key-1]]["mag"] - measurement_dict[key]["mag"]
            print "[*]",
            print key, reference_dict[sorted_rkeys[key-1]]["mag"], \
                  measurement_dict[key]["mag"], diff
            diffs.append(diff)
    print "[*] Median difference : %.3f" % np.median(diffs)
    print "[*] Standard Deviation in differences : %.3f" % np.std(diffs)

    # reject poor quality measurements
    tmp_diffs = diffs[:]
    for i, key in enumerate(measurement_dict.keys()[:]):
        if key >= num_measurements - num_check:
            continue
        if tmp_diffs[i] > np.median(tmp_diffs) + np.std(tmp_diffs) or tmp_diffs[i] < np.median(tmp_diffs) - np.std(tmp_diffs):
            id = reference_dict[sorted_rkeys[key-1]]["id"]
            print "[!] Rejecting %s with difference = %.3f." % (id, tmp_diffs[i])
            del reference_dict[sorted_rkeys[key-1]]
            del measurement_dict[key]
            diffs.remove(tmp_diffs[i])

    median_diff = np.median(diffs)
    diff_sig = np.std(diffs)
    
    print "[*] New Median difference : %.3f" % median_diff
    print "[*] New Standard Deviation in differences : %.3f" % diff_sig
    return median_diff, diff_sig, sorted_rkeys

def check_zeropoint_offset(reference_dict, measurement_dict, median_diff, diff_sig, num_check, sorted_rkeys):
    
    #num_measurements = np.max(measurement_dict.keys())
    num_measurements = len(measurement_dict.keys())
    
    print "[*] Testing zero-point against SDSS stars."
    print measurement_dict.keys()
    print len(measurement_dict.keys())
    print num_measurements
    print num_check
    for key in measurement_dict.keys():
        if key < num_measurements - num_check or key == num_measurements:
            continue
        try:
            diff = reference_dict[sorted_rkeys[key-1]]["mag"] - (measurement_dict[key]["mag"]+median_diff)
        except KeyError, e:
            print e
            continue
        error = np.sqrt(measurement_dict[key]["mag_err"]*measurement_dict[key]["mag_err"] + diff_sig*diff_sig)
        if diff + error >0 and diff - error >0:
            print "[!] %s %.3f +/- %.3f not consistent with zero-point." % (reference_dict[sorted_rkeys[key-1]]["id"], diff, error)
            continue
        if diff + error < 0 and diff - error < 0:
            print "[!] %s %.3f +/- %.3f not consistent with zero-point." % (reference_dict[sorted_rkeys[key-1]]["id"], diff, error)
            continue
        print "[+] %s %.3f +/- %.3f is consistent with zero-point." % (reference_dict[sorted_rkeys[key-1]]["id"], diff, error)
        return None

def photometry_pipeline(image, diff, path, extent, output_prefix, extension, num_check):
    imageFile = path+image
    diffFile = path+diff
    # take care of some house keeping, checking dao_params.txt exists and removing
    # files generated by previous iraf sessions that could interfere.
    house_keeping(path, imageFile, diffFile)
    
    # load image data
    data, image, hdulist, size, mid_point = load_image_data(imageFile, extent, extension=extension)
    filter = hdulist[extension].header["HIERARCH FPA.FILTERID"].split(".")[0]
    
    # get the seeing in pixels
    seeing_in_pix = hdulist[extension].header["CHIP.SEEING"]
    
    # find sources
    sources, bkg_sigma = find_sources(imageFile, data, seeing_in_pix, threshold=5.)
    
    # get positions of detected sources
    sources, positions, transient_position, transient_index = get_detection_positions(sources, extent, mid_point)
    
    # select quality detections
    quality_detections = apply_ML(data, sources)
    print
    print "[*] Quality detections: "
    for i, index in enumerate(quality_detections):
        print "   [*]",
        print i, index, sources['xcentroid'][index], sources['ycentroid'][index]
    
    # build a dictionary of quality reference stars.
    reference_dict = cross_match_sdss(imageFile, extension, sources, quality_detections, extent)

    # assert there are enough remaining stars to get a sensible measurement
    try:
        assert len(reference_dict.keys()) >= 9+num_check #DEBUGGING change this to something like 9
    except AssertionError:
        print "[!] %d reference stars is not enough for a reliable measurement." % (len(reference_dict.keys()))
        print "[!] Moving onto next image."
        return 0

    # perform PSF photometry with daophot
    zp_estimate, psfimage = doaphot_psf_photometry(path, imageFile, extent, extension)
    
    # calculate scatter in measurents compared to SDSS
    measurement_dict = {}
    for line in open(imageFile+".als.1","r").readlines():
        if "#" in line or "\\" not in line:
            continue
        data = line.rstrip().split()
        measurement_dict[int(data[0])] = {"mag":float(data[3]), "mag_err":float(data[4]), \
            "pos":(float(data[1]), float(data[2]))}

    # calculate sdss refence offset from zero point estimate (default is 25 for photpars)
    median_diff, diff_sig, sorted_rkeys = calculate_zeropoint_offset(reference_dict, measurement_dict, num_check)
    
    # check zero point offset against sdss test stars
    check_zeropoint_offset(reference_dict, measurement_dict, median_diff, diff_sig, num_check, sorted_rkeys)
    
    # calculate the zero point
    print "[+] Zero-point : %.3f +/- %.3f" % (zp_estimate+median_diff, diff_sig)
    zp_output = open("/".join(path.split("/")[:-2])+"/"+output_prefix+"_zero-point.txt", "a+")
    zp_output.write("%s %s %s %s %s\n" % (imageFile, hdulist[0].header["MJD-OBS"], \
                                          filter, zp_estimate+median_diff, diff_sig))
    zp_output.close()
                                          
    # now perform photometry on difference image
    
    # first get location of transient in diff, using intial guesstimate of location in transient_position
    diff_pos_output = open(diffFile+".coo","w")
    diff_pos_output.write("%s %s\n"%(transient_position[0], transient_position[1]))
    diff_pos_output.close()
    imx = iraf.imexam(imageFile, frame=1, use_display=0, defkey="a", imagecur=diffFile+".coo", Stdout=1)
    i = 2
    while i < len(imx):
        transient_position = (eval(imx[i].split()[0]), eval(imx[i].split()[1]))
        i = i+2
    diff_pos_output = open(diffFile+".coo","w")
    diff_pos_output.write("%s %s\n"%(transient_position[0], transient_position[1]))
    diff_pos_output.close()
    
    # perform photometry on difference image
    run_phot(diffFile, diffFile+".coo")
    run_allstar(diffFile,psfimage)
    try:
        for line in open(diffFile+".als.1","r").readlines():
            if "#" in line or "\\" not in line:
                continue
            data = line.rstrip().split()
            diff_mag = float(data[3])
            diff_mag_err = float(data[4])
            diff_pos = (float(data[1]), float(data[2]))
        
        diff_mag = diff_mag+median_diff
        diff_mag_err = np.sqrt(diff_mag_err*diff_mag_err + diff_sig*diff_sig)
        print "[+] Transient magnitude from diff: %.3f +/- %.3f" % (diff_mag, diff_mag_err)
        diff_output = open("/".join(path.split("/")[:-2])+"/"+output_prefix+"_diff_mags.txt","a+")
        diff_output.write("%s %s %s %s %s\n" % (diffFile, hdulist[0].header["MJD-OBS"], filter, diff_mag, diff_mag_err))
        diff_output.close()
    except IOError:
        print "[!] %s not found. No difference image measurements found." % (diffFile+".als.1")
        print "[*] Exiting."
        sys.exit(0)

def main():
    
    parser = optparse.OptionParser("[!] usage: python photometry_pipeline.py\n"+\
                                   "-p <path to images>\n"+\
                                   "-i <images [comma-separated list]>\n"+\
                                   "-e <extent [in pixels]>\n"+\
                                   "-o <output prefix>\n"+\
                                   "-E <fits extension>\n"+\
                                   "-n <number of reference stars to use for testing>\n"+\
                                   "-d <difference image file>"
                                   )
        
    parser.add_option("-p", dest="path", type="string", \
                      help="specify path to image file[s] to be analysed.")
    parser.add_option("-i", dest="imagesFile", type="string", \
                      help="specify file containing image names for analysis.")
    parser.add_option("-e", dest="extent", type="float", \
                      help="extent of images for analysis.")
    parser.add_option("-o", dest="output_prefix", type="string", \
                      help="sepcify the output file name.")
    parser.add_option("-E", dest="extension", type="int", \
                      help="sepcify the fits file extension to be used.")
    parser.add_option("-n", dest="num_check", type="int", \
                      help="sepcify the number of reference stars to use as check.")
    parser.add_option("-d", dest="diffsFile", type="str", \
                      help="sepcify the file containing the difference image names for analysis.")
    
    (options, args) = parser.parse_args()

    try:
        path = options.path
        imagesFile = options.imagesFile
        extent = options.extent
        output_prefix = options.output_prefix
        extension = options.extension
        num_check = options.num_check
        diffsFile = options.diffsFile
    except AttributeError, e:
        print e
        print parser.usage
        exit(0)

    required_args = [path, imagesFile, extent, output_prefix, extension, num_check, diffsFile]

    if None in required_args:
        print parser.usage
        exit(0)

    for line1 in open(path+imagesFile,"r").readlines():
        image = line1.rstrip()
        for line2 in open(path+diffsFile,"r").readlines():
            diff = line2.rstrip()
            try:
                assert image in diff
                try:
                   photometry_pipeline(image, diff, path, extent, output_prefix, extension, num_check)
                except Exception, e:
                    print e
                    print "[!] Something went wrong while performing photometry."
                    print "[*] Moving on to next image."
            except AssertionError:
                continue




if __name__ == "__main__":
    main()
