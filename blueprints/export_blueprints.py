# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:20:26 2024

@author: vkuhlen
"""

import argparse
from collections import deque
from functools import partial
import json
from pathlib import Path
from typing import Any, TypeAlias
from uuid import UUID


Blueprint: TypeAlias = dict[str, Any]


SEQUENCES_TO_EXPORT = [
  '2D radial sequence',
  '2D spiral sequence',
  '3D MP-RAGE sequence',
  '3D radial sequence',
  'Abstract sequence',
  'Demo FID sequence',
  'Demo FLASH sequence',
  'EPI music sequence',
  'EPI sequence',
  'FLASH sequence',
  'GRASE sequence',
  'RARE sequence',
  'SE EPI diffusion sequence',
  'SE EPI sequence',
  'Template sequence',
  'bSSFP sequence',
]


def loadLibrary(libraryFile: Path) -> dict[UUID, Blueprint]:
  with open(libraryFile, 'r', encoding='utf-8') as _f:
    library = json.load(_f)
    blueprints = library['blueprints']
  return blueprints
  

def findBlueprintByName(blueprints: dict[UUID, Blueprint], name: str):
  for key, bp in blueprints.items():
    if bp['name'] == name:
      return key
  raise KeyError(f'Blueprint {name} not found')
  

def collectAllUsedBlueprintIds(blueprints: dict[UUID, Blueprint],
                               blueprintId: UUID):
  usedBlueprintIds = set()
  bpIdQueue = deque([blueprintId])
  while len(bpIdQueue) > 0:
    seqBpId = bpIdQueue.pop()
    usedBlueprintIds.add(seqBpId)
  
    for definition in blueprints[seqBpId]['definitions'].values():
      if (bpId := definition.get('blueprint_id')) is not None:
        bpIdQueue.appendleft(bpId)
  return usedBlueprintIds
    
        
def writeBlueprintsToFile(outputPath: Path, blueprints: list[Blueprint]):
  outputPath.mkdir(exist_ok=True, parents=True)
  for blueprint in blueprints:
    with open(outputPath / f'{blueprint["id"]}.json', 'w', 
              encoding='utf-8', newline='\n') as _f:
      json.dump(blueprint, _f, indent=2, ensure_ascii=False)
      
def writeBlueprintIdMapping(outputPath: Path, blueprints: list[Blueprint]):
  output = ['| Name | Id  |']
  output.append('| :-- | :-: |')
  for blueprint in sorted(blueprints, key=lambda b: b['name']):
    output.append(f'| {blueprint["name"]} | {blueprint["id"]} |')
  with open(outputPath, 'w', newline='\n') as _f:
    _f.write('\n'.join(output))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Export blueprints from single database file to folder.')
  parser.add_argument('-i', '--input',
                      type=str,
                      help='Path to the library file to export from')
  parser.add_argument('-o', '--output',
                      type=str,
                      help='Path to output folder')
  args = parser.parse_args()
 
  libraryFile = Path(args.input)
  blueprints = loadLibrary(libraryFile)
  blueprintIdsToExport = [findBlueprintByName(blueprints, name)
                          for name in SEQUENCES_TO_EXPORT]

  usedBlueprintIds = set().union(*map(partial(collectAllUsedBlueprintIds,
                                              blueprints),
                                      blueprintIdsToExport))
  usedBlueprintIds.add('Atomic')
  usedBlueprints = [blueprints[bpId] for bpId in usedBlueprintIds]
  outputPath = Path(args.output)
  writeBlueprintsToFile(outputPath, usedBlueprints)
  writeBlueprintIdMapping(outputPath / 'name_to_id_mapping.md', usedBlueprints)