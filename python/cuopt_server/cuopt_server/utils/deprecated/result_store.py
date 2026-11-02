# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class ResultStore(ABC):
    def __init__(self, done_attribute):
        """
        'done_attribute' should be the name of an attribute
        in an object or dictionary that can be checked to see
        if the job is complete. The method of getting the
        value of the attribute is up to the implementation,
        as is the interpretation of values (although True or
        False would be obvious value choices.
        """
        self.done_attribute = done_attribute

    @abstractmethod
    def put(self, id, obj):
        """
        Inserts an item into the result store if the key
        'id' does not already exist. If the key exists,
        a ValueError is raised.
        """
        pass

    @abstractmethod
    def get(self, id):
        """
        Return the item stored under key 'id' if it
        exists, or None otherwise.
        """
        pass

    @abstractmethod
    def delete(self, id):
        """
        Delete the item stored under key 'id' if it exists.
        """
        pass

    @abstractmethod
    def get_and_delete_if_done(self, id):
        """
        If the key 'id' exists in the result store, return
        a tuple containing the item and the value of the done attribute.
        If the key 'id' does not exist, return (None, None).
        If the key exists and the done attribute is set (the job is
        complete), then delete the item from the result store.
        """
        pass

    @abstractmethod
    def update(self, id, obj):
        """
        If the key 'id' exists, update the item with obj.
        If the key 'id' does not exist, insert obj.
        """
        pass
