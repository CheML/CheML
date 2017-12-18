"""Dataset loader helpers. This file collects everything related to downloading
and organizing CheML and other open datasets for computational chemistry."""
import os
import numpy as np
import sys
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
except:
    from urllib2 import urlopen, HTTPError, URLError
import pickle
from sklearn.datasets.base import Bunch
from scipy.io import loadmat
from sklearn.decomposition import PCA
import tarfile


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
    HF2_1K=("HF/HF2_1K.pkl", HF_URL_BASE + "data_m2.pkl"),
    HF3_1K=("HF/HF3_1K.pkl", HF_URL_BASE + "data_m3.pkl"),
    HF4_1K=("HF/HF4_1K.pkl", HF_URL_BASE + "data_m4.pkl"),
    HF5_1K=("HF/HF5_1K.pkl", HF_URL_BASE + "data_m5.pkl"),
    HF6_1K=("HF/HF6_1K.pkl", HF_URL_BASE + "data_m6.pkl"),
    HF2_7K=("HF/HF2_7K.pkl", HF_URL_BASE + "data_m2_7k.pkl"),
    HF3_10K=("HF/HF3_10K.pkl", HF_URL_BASE + "data_m3_10k.pkl"),
    HF4_10K=("HF/HF4_10K.pkl", HF_URL_BASE + "data_m4_10k.pkl"),
    HF5_10K=("HF/HF5_10K.pkl", HF_URL_BASE + "data_m5_10k.pkl"),
    HF6_10K=("HF/HF6_10K.pkl", HF_URL_BASE + "data_m6_10k.pkl"),
    HX2=("HF/HX2.pkl", HF_URL_BASE + "data_HX2.pkl"),
    HX3=("HF/HX3.pkl", HF_URL_BASE + "data_HX3.pkl"),
    HX4=("HF/HX4.pkl", HF_URL_BASE + "data_HX4.pkl"),
    HX5=("HF/HX5.pkl", HF_URL_BASE + "data_HX5.pkl"),
    HX6=("HF/HX6.pkl", HF_URL_BASE + "data_HX6.pkl"),
    QM7=("GDB13/qm7.mat", "http://quantum-machine.org/data/qm7.mat"),
    QM9=("GDB13/qm9.pkl", "https://ndownloader.figshare.com/files/7003292"),
    QM9_bonds=("GDB13/qm9_bonds.npz",
                "https://berkeley.box.com/shared/static/"
                "2mq7cgd8aypqy7js1mr8lztkn8ky34qj.npz"),
    QM9_properties=("GDB13/qm9_properties.npy",
        "https://berkeley.box.com/shared/static/"
        "ptjxewyrg1ahcs2lkr6fq1jyuyfnoemo.npy")
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
            if not os.path.exists(dirname):
                os.makedirs(dirname)
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
    except URLError as e:
        raise
    except HTTPError as e:
        raise


def _tar_decompress(local_name, filename):

    tar = tarfile.open(local_name)

    # Assume single file member in tar file
    members = tar.getnames()

    if len(members) == 1:
        content = tar.extractfile(members[0])
        with open(filename, 'wb') as local_file:
            local_file.write(content.read())
    else:
        tar.extractall(filename)

    os.remove(local_name)


def _get_or_download_dataset(dataset_name, path=None, suffix=None):
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
        if suffix is not None:
            local_name = filename + suffix
        else:
            local_name = filename
        print("Downloading {} to {}...".format(url, local_name))
        _download(url, local_name)
        print("... done.")

        if suffix is not None:
            if 'tar.gz' in suffix:
                print("Decompress and tar file {}...".format(local_name))
                _tar_decompress(local_name, filename)
                print("... done.")
        return filename


def _open_pickle(filename):
    # hack from http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    # Needs to be extensively tested between versions


    with open(filename, 'rb') as f:
        try:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
        except AttributeError:
            p = pickle.load(f)
    return Bunch(**p)


def load_HF2(path=None, large=False):
    dataset_name = 'HF2_7K' if large else 'HF2_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HF3(path=None, large=False):
    dataset_name = 'HF3_10K' if large else 'HF3_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HF4(path=None, large=False):
    dataset_name = 'HF4_10K' if large else 'HF4_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HF5(path=None, large=False):
    dataset_name = 'HF5_10K' if large else 'HF5_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HF6(path=None, large=False):
    dataset_name = 'HF6_10K' if large else 'HF6_1K'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)


def load_HX2(path=None):
    filename = _get_or_download_dataset('HX2', path=path)
    return _open_pickle(filename)

def load_HX3(path=None):
    dataset_name = 'HX3'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HX4(path=None):
    dataset_name = 'HX4'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HX5(path=None):
    dataset_name = 'HX5'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def load_HX6(path=None):
    dataset_name = 'HX6'
    filename = _get_or_download_dataset(dataset_name, path=path)
    return _open_pickle(filename)

def _gdb_align(bunch, align, only_planar, planarity_tol):
    sys.stdout.flush()
    pca = PCA()
    keep_molecule = []
    has_bonds = "O" in bunch
    for c,(positions, charges) in enumerate(zip(bunch.R, bunch.Z)):
        transformed = np.vstack([
            pca.fit_transform(positions[charges != 0]),
            np.zeros([(charges == 0).sum(), 3])])
        if has_bonds:
            transformed_bonds=np.vstack([pca.transform(bunch.B[c,bunch.O[c]!=0].reshape((-1,3))).reshape((-1,2,3)),
                np.zeros([(bunch.O[c] == 0).sum(),2 , 3])])


        # the following evaluates how much variance is in the first two axes
        # before this, the algorithm was also using zero positions, leading
        # to 454 planar molecules (for QM7):
        # pca.fit(positions).explained_variance_ratio_[:2].sum() #  
        # currently, the algorithm yields 415 planar molecules
        var_2D = pca.explained_variance_ratio_[:2].sum()
        keep = (not only_planar) or var_2D > 1 - planarity_tol
        keep_molecule.append(keep)
        if align and keep:
            positions[:] = transformed
            if has_bonds:
                bunch.B[c,:] = transformed_bonds

    return keep_molecule

def load_qm7(path=None, align=False, only_planar=False, planarity_tol=.01):
    # import ipdb;ipdb.set_trace()
    filename = _get_or_download_dataset("QM7", path=path)
    qm7_file = loadmat(filename)
    qm7_bunch = Bunch(**{k:v for k, v in qm7_file.items()
        if k in ['P', 'X', 'T', 'Z', 'R']})
    if align or only_planar:
        keep_molecule = _gdb_align(qm7_bunch, align, only_planar, planarity_tol)
           
        if only_planar:
            keep_molecule = np.array(keep_molecule)
            qm7_bunch['X'] = qm7_bunch.X[keep_molecule]
            qm7_bunch['T'] = qm7_bunch.T[:, keep_molecule].ravel()
            qm7_bunch['Z'] = qm7_bunch.Z[keep_molecule]
            qm7_bunch['R'] = qm7_bunch.R[keep_molecule]

            new_molecule_indices = -np.ones_like(keep_molecule, dtype='int')
            new_molecule_indices[keep_molecule] = np.arange(keep_molecule.sum())
            P = [new_molecule_indices[p[keep_molecule[p]]]
                    for p in qm7_bunch['P']]
            qm7_bunch['P'] = P
            CV = []
            for i in range(len(P)):
                train = np.concatenate([p for j, p in enumerate(P) if j != i])
                test = P[i]
                CV.append((train, test))
            qm7_bunch['CV'] = CV

    return qm7_bunch

def load_qm9(path=None, align=False, only_planar=False, planarity_tol=.01):
    # import ipdb;ipdb.set_trace()
    filename = _get_or_download_dataset("QM9", path=path, suffix='.tar.gz')
    filename_bonds = _get_or_download_dataset("QM9_bonds", path=path, 
                                                    suffix=None)
    filename_properties = _get_or_download_dataset("QM9_properties",
            path=path, suffix=None)

    qm9_file = _open_pickle(filename)
    qm9_bonds = np.load(filename_bonds)
    qm9_file['R'] = qm9_file['xyz']
    qm9_file['B'] = qm9_bonds['bond_atoms']
    qm9_file['O'] = qm9_bonds['bond_orders']
    qm9_file['T'] = qm9_file['E']
    qm9_bunch = Bunch(**{k:v for k, v in qm9_file.items()
        if k in ['R', 'Z', 'T','B','O']})

    qm9_properties = np.load(filename_properties)
    property_names = qm9_properties.dtype.names
    qm9_bunch.update(**{p:qm9_properties[p] for p in property_names
                if p not in ('A', 'B', 'C')})
    qm9_bunch.update({f'{a}rot': qm9_properties[a] for a in ('A', 'B', 'C')})

    if align or only_planar:
        keep_molecule = _gdb_align(qm9_bunch, align, only_planar, planarity_tol)

        if only_planar:
            keep_molecule = np.array(keep_molecule)
            qm9_bunch['T'] = qm9_bunch.T[keep_molecule]
            qm9_bunch['Z'] = qm9_bunch.Z[keep_molecule]
            qm9_bunch['R'] = qm9_bunch.R[keep_molecule]
            qm9_bunch['B'] = qm9_bunch.B[keep_molecule]
            qm9_bunch['O'] = qm9_bunch.O[keep_molecule]
    return qm9_bunch


