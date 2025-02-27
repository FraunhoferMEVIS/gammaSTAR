# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:00:15 2025

@author: vkuhlen
"""

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias
from uuid import UUID


Blueprint: TypeAlias = dict[str, Any]


def saveLibrary(libraryFile: Path, blueprints: dict[UUID, Blueprint]):
  with open(libraryFile, 'w', encoding='utf-8', newline='\n') as _f:
    json.dump({'blueprints': blueprints}, _f,  indent=2, 
              ensure_ascii=False, sort_keys=True)


def importBlueprintsFromFolder(importPath: Path) -> dict[UUID, Blueprint]:
  blueprints = {}
  for blueprintFile in filter(lambda f: f.suffix == '.json',
                              importPath.iterdir()):
    with open(blueprintFile, 'r', encoding='utf-8') as _f:
      blueprint = json.load(_f)
    blueprints[blueprint['id']] = blueprint
  return blueprints

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Export blueprints from single database file to folder.')
  parser.add_argument('-i', '--input',
                      type=str,
                      help='Path to folder to import blueprints from')
  parser.add_argument('-o', '--output',
                      type=str,
                      help='Path to new database file')
  args = parser.parse_args()
  
  libraryFile = Path(args.output)
  importPath = Path(args.input)
  
  blueprintsToImport = importBlueprintsFromFolder(importPath)

  saveLibrary(libraryFile, blueprintsToImport)