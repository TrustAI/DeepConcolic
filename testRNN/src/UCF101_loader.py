import urllib.request
import os
import sys
import patoolib

URL_LINK = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_ucf(data_dir_path):
    ucf_rar = data_dir_path + '/UCF101.rar'

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    if not os.path.exists(ucf_rar):
        print('ucf file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=URL_LINK, filename=ucf_rar,
                                   reporthook=reporthook)

    print('unzipping ucf file')
    patoolib.extract_archive(ucf_rar, outdir=data_dir_path)


def scan_ucf(data_dir_path, limit):
    input_data_dir_path = data_dir_path + '\\UCF-101'

    result = dict()

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = f
        if dir_count == limit:
            break
    return result


def scan_ucf_with_labels(data_dir_path, labels):
    input_data_dir_path = data_dir_path + '\\UCF-101'

    result = dict()

    dir_count = 0
    for label in labels:
        file_path = input_data_dir_path + os.path.sep + label
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = label
    return result



def load_ucf(data_dir_path):
    UFC101_data_dir_path = data_dir_path + "\\UCF-101"
    if not os.path.exists(UFC101_data_dir_path):
        download_ucf(data_dir_path)


def main():
    data_dir_path = '../very_large_data'
    load_ucf(data_dir_path)


if __name__ == '__main__':
    main()
