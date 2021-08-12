# SMART_ID

Installation
Requirements
Python 3 needs to be installed. Install Python 3 with Anaconda

Once Python 3 is installed, Clone this repository and then create a conda environment

conda create -n snart_id python=3.8
conda activate snart_id
conda install -c anaconda pyaudio
pip install deepface
pip install -r requirements.txt
You also need to install ffmpeg.

If you are on Linux then run below command in the terminal

sudo apt update
sudo apt install ffmpeg
If you are on Windows, Then download ffmpeg package for windows and then extract it. Once extracted, Add the path of bin folder to Environment Variable in Windows.

Now the installation if done.

How to run the project
Go to the project directory and make sure conda environment is active. If not then you can activate by running below command

conda activate snart_id


Now start server by running

python app.py
Now the server will start. You can ctrl+click on the link in the terminal you can paste http://localhost:5000/ in your browser to launch the site.
