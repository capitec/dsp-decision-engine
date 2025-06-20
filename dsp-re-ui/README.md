# Data Science Platform (DSP) Rule Engine UI

## Overview

The DSP Rule Engine UI is a React-based application that allows users to visually build decision trees for rule-based decision-making systems. Once the trees are built, users can save and execute them through endpoints or visualize and test them within a Jupyter notebook.

## Prerequisites

Before getting started, make sure you have the following installed:

- **Node.js** (for running the React app)
- **Yarn** or **npm** (for managing dependencies)
- **Python** (for running the backend model and Jupyter notebooks)
- **Graphviz** (for visualizing decision trees in the notebook)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/capitec/dsp-re-ui.git
    cd dsp-re-ui
    ```

2. Install the dependencies using **Yarn** or **npm**:

    - Using **npm**:
        ```bash
        npm install
        ```

    - Or using **Yarn** (if you prefer Yarn):
        ```bash
        yarn install
        ```

## Running the React App

To start the React application in development mode:

- Using **npm**:
    ```bash
    npm start
    ```

- Using **Yarn**:
    ```bash
    yarn start
    ```

This will start the UI on `http://localhost:3000`, where you can interact with the decision tree builder.

## Visual Decision Tree Builder

The UI allows you to visually build decision trees. Once a decision tree is created, you can save it to a configuration file.

1. **Add Config to `config.json`**:
   - After building the tree in the UI, save the configuration to the file `dsp-re-ui/example/tree/source_dir/config.json`.
   
2. **Run the Model with VSCode**:
   - You can use VSCode to launch the decision tree model as an endpoint based on the config file.

3. **Visualize and Test in Jupyter Notebook**:
   - Alternatively, you can use the notebook `dsp-re-ui/example/tree/test.ipynb` to visualize the decision tree (make sure **Graphviz** is installed) and test it with batch data on the model.

## Running the Project Manually

If you'd prefer to run the project manually, you can use the following command:

```bash
MODEL_PREFIX="dsp-re-ui/example/tree" MODEL_RELATIVE_PATH="source_dir" MODEL_VERSION="0.0.0" uvicorn spockflow.inference.server.asgi:app --reload
```

Make sure to replace the relevant paths if needed, and this will start the backend server for your decision tree model.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

Now, the README includes options for **Yarn** alongside the npm instructions, offering flexibility for users who prefer Yarn as their package manager.