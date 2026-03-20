FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /sandbox

# Install required mathematical and neural network packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install transformers sympy z3-solver

# Copy the source code
COPY . .

# Force strict NVIDIA hardware allocations for Multi-Tenant MIG splits
ENV NVIDIA_VISIBLE_DEVICES=all
ENV ISOLATED_SANDBOX=true

# Entry point sets up isolated Execution environment for EBM
CMD ["python", "-c", "import time; print('Sandbox Initialized'); time.sleep(86400)"]
