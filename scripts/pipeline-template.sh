#!/bin/bash

# m4_ignore(
echo "This is just a script template, not the script (yet) - pass it to 'argbash' to fix this." >&2
exit 11  #)Created by argbash-init v2.10.0
# ARG_OPTIONAL_SINGLE([GPU])
# ARG_POSITIONAL_SINGLE([input], i, [Input directory containing images])
# ARG_POSITIONAL_SINGLE([output], o, [Output directory with images and pagexml])
# ARG_DEFAULTS_POS
# ARG_HELP([<The general help message of my script>])
# ARGBASH_GO

# [ <-- needed because of Argbash

# vvv  PLACE YOUR CODE HERE  vvv
# For example:
printf 'Value of --%s: %s\n' 'GPU' "$_arg_gpu"
printf "Value of '%s': %s\\n" 'input' "$_arg_input"
printf "Value of '%s': %s\\n" 'output' "$_arg_output"

# ^^^  TERMINATE YOUR CODE BEFORE THE BOTTOM ARGBASH MARKER  ^^^

# ] <-- needed because of Argbash
