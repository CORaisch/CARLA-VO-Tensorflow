#!/bin/bash

# clear predictions
if [ -d "predictions" ]; then
    rm -rf predictions/*
else
    mkdir predictions
fi
