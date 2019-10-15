#!/bin/bash

# clear csv logs
if [ -d "logs/csv" ]; then
    rm -rf logs/csv/*
else
    mkdir -p logs/csv
fi

# clear tensorboard logs
if [ -d "logs/tensorboard" ]; then
    rm -rf logs/tensorboard/*
else
    mkdir -p logs/tensorboard
fi

# clear checkpoints
if [ -d "checkpoints" ]; then
    rm -rf checkpoints/*
else
    mkdir checkpoints
fi
