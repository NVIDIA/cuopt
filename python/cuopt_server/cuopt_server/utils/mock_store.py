# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from cuopt_server.utils.result_store import ResultStore


class MockStore(ResultStore):
    def __init__(self, done_attribute):
        """
        'done_attribute' should be the name of an attribute
        in an object or dictionary that can be checked to see
        if the job is complete. The method of getting the
        value of the attribute is up to the implementation,
        as is the interpretation of values (although True or
        False would be obvious value choices.
        """
        super().__init__(done_attribute)
        self.results = {}

    def put(self, id, obj):
        """
        Inserts an item into the result store if the key
        'id' does not already exist. If the key exists,
        a ValueError is raised.
        """
        if id in self.results:
            raise ValueError(f"duplicate key {id}")
        logging.info(f"result_store adding key {id}")
        self.results[id] = obj

    def get(self, id):
        """
        Return the item stored under key 'id' if it
        exists, or None otherwise.
        """
        if id in self.results:
            return self.results[id]
        return None

    def delete(self, id):
        """
        Delete the item stored under key 'id' if it exists.
        """
        logging.info(f"result_store deleting {id}")
        self.results.pop(id, None)

    def get_and_delete_if_done(self, id):
        """
        If the key 'id' exists in the result store, return
        a tuple containing the item and the value of the done attribute.
        If the key 'id' does not exist, return (None, None).
        If the key exists and the done attribute is set (the job is
        complete), then delete the item from the result store.

        """
        logging.info(f"result_store looking for key {id}")
        if id in self.results:
            logging.info(f"result_store key present {id}")
            if not self.results[id][self.done_attribute]:
                logging.info(f"returning not done for {id}")
                return self.results[id], False
            logging.info(f"returning done for {id} and deleting")
            return self.results.pop(id), True
        else:
            logging.info(f"result_store key not present {id}")
        return None, None

    def update(self, id, obj):
        """
        If the key 'id' exists, update the item with obj.
        If the key 'id' does not exist, insert obj.
        """
        logging.info(f"result_store updating key {id}")
        self.results[id] = obj
