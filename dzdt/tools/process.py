#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2024 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2024	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Any, Generic, List, TypeVar

T = TypeVar("T")

class SeqProcessor(Generic[T]):
    def __init__(self) -> None:
        self.init()
    def init(self) -> None:
        pass
    def next(self, e: T) -> List[T]:
        raise NotImplementedError()
    def stop(self) -> List[T]:
        return []
    

class SeqPipeline:
    def __init__(self, processors: List[SeqProcessor]) -> None:
        self.processors = processors

    def init(self) -> None:
        for processor in self.processors:
            processor.init()

    def process(self, elements: List[Any]) -> List[Any]:
        self.init()
        result = []
        # buffer = [].extend(elements)
        for e in elements:
            # element = buffer.pop()
            buffer = [e]
            for processor in self.processors:
                new_buffer = []
                for e in buffer:
                    outputs = processor.next(e)
                    new_buffer = new_buffer + outputs
                buffer = new_buffer
            result = result + buffer

        buffer = []
        for processor in self.processors:
            if not buffer:
                buffer = processor.stop()
                continue
            new_buffer = []
            for e in buffer:
                outputs = processor.next(e)
                new_buffer = new_buffer + outputs
            buffer = new_buffer + processor.stop()
        result = result + buffer 

        return result


