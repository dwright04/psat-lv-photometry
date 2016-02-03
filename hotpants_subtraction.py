import sys, subprocess, os.path
import numpy as np
from astropy.io import fits
from pyraf import iraf
from cStringIO import StringIO # for Capturing
from photometry_pipeline import extract_dao_params

class Capturing(list):
    """
        Taken from:
        http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
        
        this is a context manager for capturing print statements as elements of a list.
        
        Usage:
        
        with Capturing(output) as output:
            do_something(my_object)
        """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

def get_imstats(image):
    output = []
    with Capturing(output) as output:
        iraf.imstat(image)
    output = output[1].split("   ")
    print output
    max = float(output[-1])
    min = float(output[-2])
    return max, min

def round_down(num, divisor):
    return num - (num%divisor)

def round_up(num, divisor):
    return int(np.ceil(num / divisor)) * divisor

def perform_subtraction(input, template_dict, path, bump):
    log = open(path+"log.txt","a+")
    iraf.images() # load iraf images package for imstat
    for inim in input:
        if os.path.exists("%sdiff_%s"%(path,inim.split("/")[-1])):
            print "[*] %sdiff_%s already exists." % (path,inim.split("/")[-1])
            continue
        print inim
        hdulist = fits.open(inim)
        filter = hdulist[0].header["HIERARCH FPA.FILTER"][0]
        tmplim = template_dict[filter]
        try:
            dao_params = extract_dao_params(inim.strip(".fits")+"_dao_params.txt")
        except IOError:
            print "[-] params file does not exist!"
            continue

        sky = dao_params[0]

        print get_imstats(inim), get_imstats(tmplim)

        iu, il = get_imstats(inim)
        offset = round_up(np.abs(il), 500)

        if bump:
            try:
                iraf.imarith(operand1=inim, op="+", operand2=offset, result=inim.strip(".fits")+"_p%s.fits" % (str(offset)))
            except:
               pass
            inim = inim.strip(".fits")+"_p%s.fits" % (str(offset))
    
        hdulist = fits.open(tmplim)
        tu, tl = get_imstats(tmplim)

        offset = round_up(np.abs(tl), 2000)

        if bump:
            try:
                iraf.imarith(operand1=tmplim, op="+", operand2=offset, result=tmplim.strip(".fits")+"_p%s.fits" % (str(offset)))
            except:
                pass
            tmplim = tmplim.strip(".fits")+"_p%s.fits" % (str(offset))

    
        print get_imstats(inim), get_imstats(tmplim)

        tu, tl = get_imstats(tmplim)
        iu, il = get_imstats(inim)

        #tl = round_down(tl,100)
        #il = round_down(tl,10)
        
        #tu = round_up(tu,1000)
        #iu = round_up(iu,100)

        #cmd = "hotpants -n i -c t -inim %s -tmplim %s -outim %sdiff_%s -tu %d -iu %d" % (inim, tmplim, path, inim.split("/")[-1], tu, iu)

        #cmd = "hotpants -n i -c t -inim %s -tmplim %s -outim %sdiff_%s -tu %f -tl %f -iu %f" % (inim, tmplim, path, inim.split("/")[-1], tu, tl, iu)

        #cmd = "hotpants -n i -c t -inim %s -tmplim %s -outim %sdiff_%s -tu %d -tl %d -iu %d -il %.3f" % (inim, tmplim, path, inim.split("/")[-1], tu, tl, iu, il)

        cmd = "hotpants -n i -c t -inim %s -tmplim %s -outim %sdiff_%s -tl %d -il %.3f" % (inim, tmplim, path, inim.split("/")[-1], tl, il)

        log.write(cmd+"\n\n")
        print cmd
        subprocess.call(cmd, shell=True)
        print
    log.close()

def main(argv=None):

    if argv is None:
        argv = sys.argv

    if len(argv) != 4 :
        sys.exit("Usage: python hotpants_subtraction.py <txt file contianing template names> <txt file containing target image names> <bump up the counts>")
    tmpl_file = argv[1]
    tgt_file = argv[2]
    bump = bool(int(argv[3]))
    print bump
    path_to_templates = tmpl_file.replace(tmpl_file.split("/")[-1],"")
    path_to_targets = tgt_file.replace(tgt_file.split("/")[-1],"")
    template_dict = {}
    for line in open(tmpl_file,"r").readlines():
        file = line.rstrip()
        hdulist = fits.open(path_to_templates+file)
        filter = hdulist[0].header["HIERARCH FPA.FILTER"][0]
        template_dict[filter] = path_to_templates + line.rstrip()

    tgt_list = []
    for line in open(tgt_file,"r").readlines():
        tgt_list.append(path_to_targets + line.rstrip())
    print tgt_list
    perform_subtraction(tgt_list, template_dict, path_to_targets, bump)

if __name__ == "__main__":
    main()
