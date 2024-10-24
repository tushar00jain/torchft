#!/bin/bash

set -e

FILES="$(rg -F 'Meta Platforms' --files-without-match --type rust --type python || true)"

# exit 1 if there are files without the match and print them.
if [ -n "$FILES" ]; then
  echo "Found files without header:"

  for file in $FILES;
  do
    echo $file
  done

  exit 1
fi

exit 0
