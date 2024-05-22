<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">ASL Character Classifcation: An Exploration of Machine Learning Techniques</h3>

  <p align="center">
    <em> Completed by Ashley Hart & Darian Jennings</em>
    <br />
    <p align="center">
    <a href="https://drive.google.com/file/d/1J0qYfevt7Pvixd0Xkb7pQzB65lOFgbic/view?usp=sharing"><strong>Read the Code Report »</strong></a>
    <br />
    <br />
    <a href="#usage">View Demo</a>
    ·
    <a href="https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-fml_party/issues">Report Bug</a>
    ·
    <a href="https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-fml_party/issues">Request Feature</a>
    </p>
  </p>
</p>

<p align="center">
<img width="638" alt="asl_examples" src="https://github.com/darianjennings1/ASLclassification/assets/59739081/b8234f5d-8642-4ce1-b209-a8ae0f2fb1d9">
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This work was completed as our final project submission for EEL5840 - Fundamentals of Machine Learning course for the Fall 2023 semester taught by Dr. Catia Silva. 

We developed a transfer learning model in PyTorch that can classify the first nine ASL characters in addition to attempting to label unfamiliar inputs as "unknown".

More details about our development process and the overall implementation of the model can be found in our code report lined below. 

[Code Report](https://drive.google.com/file/d/1J0qYfevt7Pvixd0Xkb7pQzB65lOFgbic/view?usp=sharing)

You can include tables or images to summarize your results when and if appropriate.

<!-- GETTING STARTED -->
## Getting Started

To run this code, you will need to have a system that is set up with Anaconda and Python 3.8 or higher. Please refer to the next two sections for details about packages you need to install. 

**NOTE: If you are working on Hipergator (HPG), it is far easier to clone the repository onto HPG and to use the kernel titled `Pytorch-2.0.1`.**

### Dependencies

Please ensure that the following dependencies are installed in the Anaconda environment you will be using to run the model.
* PyTorch w/ Torchvision
  ```sh
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch
  ```

* NumPy
  ```sh
  conda install -c conda-forge numpy
  ```

* Scikit Learn
  ```sh
  conda install -c anaconda scikit-learn
  ```

* TDQM (Progress Bar Package - Used in Model Output)
  ```sh
  conda install -c conda-forge tqdm
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-fml_party
   ```
2. Activate a conda environment that supports the dependencies listed in the section above. And you are ready to go! 

<!-- USAGE EXAMPLES -->
## Usage

Please see the train.py and test.py files to train and test our model respectively. 

**PLEASE NOTE:** Upon execution, the code will ask you to provide a path to neccessary files for the train and test scripts.

### IMPORTANT: This code must be run in the PyTorch 2.0.1 environment provided on HiPerGator.

* `train.py` - **provide a directory path to the two `.npy` files that contain the data and labels that you want to run the model on.** You can modify the lines in train.py directly OR you can provide them to the program when prompted. Please note that this script does output plots of the loss and accuracy learning curves of the model in addition to the standard training outputs. The script will also save the model to the location specified by the`save_model_path` variable. Be sure to specify a `save_model_path` if you want the trained model to be stored in a specific location. **Hyperparameters such as learning rate and weight decay can also be adjusted in this script.**

* `test.py` - provide a directory path to the one `.npy` file that contain the data that you want to run the model on. You can modify the lines in train.py directly OR you can provide them to the program when prompted. The script will load the model from the location specified by the `load_model_path` variable and output standard evaluation outputs and useful metrics. Please ensure that your `load_model_path` variable is correct prior to running the script.

### Backup Instructions
**NOTE:** If these scripts give you trouble please utilize the instructions below:

  1. Create a new Jupyter Notebook (ex: temp.ipynb)
  2. Create a single empty cell containing the following `%run training.py`, provide filepaths to your `.npy` files with the training dataset and labels when prompted or by direcly modifying the first few lines in the script.'
  4. Update the `save_model_path` variable to the directory that you want your model to be saved to. 
  5. Run the cell to train the model.
  6. Update the `load_model_path` to the location where your model was saved to by the `training.py`
  7. Create a second cell that contains the following `%run test.py`, provide a single filepath to your `.npy` file for the testing dataset when prompted or by direcly modifying the first few lines in the script. 
  8. Run the cell to test the model.

  A PDF of the full notebook has also been included for anyone who wants to take a look at our full train-test pipeline.

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/catiaspsilva/README-template/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**. Please inform the authors if you would like to make any contributions to this project. We would be happy to connect with you. 

Follow the steps below to make a contribution.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- Authors -->
## Authors

Ashley Hart - ashley.hart@ufl.edu

Darian Jennings - darian.jennings@ufl.edu

Project Link: [FML Party - Final Project Code Repository](https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-fml_party)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Dr. Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
* Joseph Arthur Conroy
* Raul Valle
* Emma Drobina
* Jean Louis
* Dr. Juan E. Gilbert

## Thank you for taking time to check out our work!

