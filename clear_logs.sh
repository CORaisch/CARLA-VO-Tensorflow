#!/bin/bash

if [ -d "logs/csv" ]; then
    rm -rf logs/csv/*
else
    mkdir -p logs/csv
fi

if [ -d "logs/tensorboard" ]; then
    rm -rf logs/tensorboard/*
else
    mkdir -p logs/tensorboard
fi
