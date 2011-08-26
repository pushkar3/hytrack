Hybrid Tracking for OpenCV
-------------------------
Usage

    cmake . 
    make
    ./hytrack live

Or, for benchmark testing:

    ./hytrack <filename>

To download test data

    wget http://www.iai.uni-bonn.de/~kleind/tracking/datasets/seqG.zip
    unzip seqG.zip -d ./seqG
    ffmpeg -i seqG/Vid_G_rubikscube.avi seqG/%04d.png
    ./hytrack seqG/Vid_G_rubikscube.txt

