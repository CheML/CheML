"""Dataset loader helpers. This file collects everything related to downloading
and organizing CheML and other open datasets for computational chemistry."""
import os
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
except:
    from urllib2 import urlopen, HTTPError, URLError
import pickle
from sklearn.datasets.base import Bunch


def get_data_dirs(data_dir=None):
    """Returns a priority list of folders where to search for a dataset.
    
    If `data_dir` is specified, this will have highest priority. The list is
    as follows:

    1. data_dir if specified
    2. the environment variable CHEML_SHARED_DATA if specified
    3. the environment variable CHEML_DATA if specified
    4. $HOME/CheML_data
    """

    paths = []


    cheml_shared_data = os.environ.get("CHEML_SHARED_DATA", None)
    cheml_data = os.environ.get("CHEML_DATA", None)
    home_data_folder = os.path.expanduser("~/cheml_data")

    if data_dir is not None:
        paths.append(data_dir)
    if cheml_shared_data is not None:
        paths.append(cheml_shared_data)
    if cheml_data is not None:
        paths.append(cheml_data)
    paths.append(home_data_folder)

    return paths



HF_URL_BASE = ("https://raw.githubusercontent.com/SamKChang/"
                "QM_wavelet/master/data/")

dataset_info = dict(
    HF2=("HF/HF2.pkl", HF_URL_BASE + "data_m2.pkl"),
    HF3_1K=("HF/HF3_1K.pkl", HF_URL_BASE + "data_m3.pkl"),
    HF4_1K=("HF/HF4_1K.pkl", HF_URL_BASE + "data_m4.pkl"),
    HF5_1K=("HF/HF5_1K.pkl", HF_URL_BASE + "data_m5.pkl"),
    HF6_1K=("HF/HF6_1K.pkl", HF_URL_BASE + "data_m6.pkl"),
    HF3_10K=("HF/HF3_10K.pkl", HF_URL_BASE + "data_m3_10k.pkl"),
    HF4_10K=("HF/HF4_10K.pkl", HF_URL_BASE + "data_m4_10k.pkl"),
    HF5_10K=("HF/HF5_10K.pkl", HF_URL_BASE + "data_m5_10k.pkl"),
    HF6_10K=("HF/HF6_10K.pkl", HF_URL_BASE + "data_m6_10k.pkl"),
    HX2=("HF/HX2.pkl", HF_URL_BASE + "data_HX2.pkl"),
    HX3=("HF/HX3.pkl", HF_URL_BASE + "data_HX3.pkl"),
    HX4=("HF/HX4.pkl", HF_URL_BASE + "data_HX4.pkl"),
    HX5=("HF/HX5.pkl", HF_URL_BASE + "data_HX5.pkl"),
    HX6=("HF/HX6.pkl", HF_URL_BASE + "data_HX6.pkl")
    )


#def https_open_with_auth(url, user, passwd):
#    request = urllib2.Request(url)
#    user_pass = base64.b64encode('{}:{}'.format(user, passwd))
#    request.add_header("Authorization", "Basic {}".format(user_pass))
#    return urllib2.urlopen(request)


def _find_file(paths, filename):

    abs_paths = [os.path.join(path, filename) for path in paths]
    for filepath in abs_paths:
        if os.path.exists(filepath):
            return filepath
    return None


def _get_first_writeable_path(paths, filename):

    abs_paths = [os.path.join(path, filename) for path in paths]
    dirs = [os.path.dirname(filepath) for filepath in abs_paths]
#    basenames = [os.path.basename(filepath) for filepath in abs_paths]
    errors = []
    for dirname, filename in zip(dirs, abs_paths):
        try:
            os.makedirs(dirname, exist_ok=True)
            existed = os.path.exists(filename)
            with open(filename, 'a'):
                pass
            if not existed:
                os.remove(filename)
            return filename
        except Exception as e:
            errors.append(e)
    raise OSError(
            "CheML could not store in any of the following directories:\n\n" +
            "\n".join(dirs))


def _download(url, filename):

    try:
        f = urlopen(url)
        with open(filename, 'wb') as local_file:
            local_file.write(f.read())
    except urllib2.URLError as e:
        raise
    except urllib2.HTTPError as e:
        raise


def _get_or_download_dataset(dataset_name, path=None):
    rel_path, url = dataset_info[dataset_name]
    
    if path is None:
        paths = get_data_dirs()
    else:
        paths = [path]
    filename = _find_file(paths, rel_path)
    if filename is not None:
        return filename
    else:
        filename = _get_first_writeable_path(paths, rel_path)
        print("Downloading {} to {}...".format(url, filename))
        _download(url, filename)
        print("... done.")
        return filename


def _open_HF_pickle(filename):
    # hack from http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    # Needs to be extensively tested between versions


    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return Bunch(**p)


def load_HF2(path=None):
    filename = _get_or_download_dataset('HF2', path=path)
    return _open_HF_pickle(filename)

def load_HF3(path=None, large=False):
    dataset_name = 'HF3_10K' if large else 'HF3_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_HF_pickle(filename)

def load_HF4(path=None, large=False):
    dataset_name = 'HF4_10K' if large else 'HF4_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_HF_pickle(filename)

def load_HF5(path=None, large=False):
    dataset_name = 'HF5_10K' if large else 'HF5_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_HF_pickle(filename)

def load_HF6(path=None, large=False):
    dataset_name = 'HF6_10K' if large else 'HF6_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_HF_pickle(filename)



