#!/usr/bin/env python
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys

import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def test_infer(
    model_name,
    x_data,
    headers=None,
    request_compression_algorithm=None,
    response_compression_algorithm=None,
):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("X", [1, 4], "FP64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(x_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput("label", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("probabilities", binary_data=False))
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm,
    )

    return results


def test_infer_no_outputs(
    model_name,
    x_data,
    headers=None,
    request_compression_algorithm=None,
    response_compression_algorithm=None,
):
    inputs = []
    inputs.append(httpclient.InferInput("X", [1, 4], "FP64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(x_data, binary_data=False)


    results = triton_client.infer(
        model_name,
        inputs,
        outputs=None,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable encrypted link to the server using HTTPS",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        required=False,
        default=None,
        help="File holding client private key. Default is None.",
    )
    parser.add_argument(
        "--cert-file",
        type=str,
        required=False,
        default=None,
        help="File holding client certificate. Default is None.",
    )
    parser.add_argument(
        "--ca-certs",
        type=str,
        required=False,
        default=None,
        help="File holding ca certificate. Default is None.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        required=False,
        default=False,
        help="Use no peer verification in SSL communications. Use with caution. Default is False.",
    )
    parser.add_argument(
        "-H",
        dest="http_headers",
        metavar="HTTP_HEADER",
        required=False,
        action="append",
        help="HTTP headers to add to inference server requests. "
        + 'Format is -H"Header:Value".',
    )
    parser.add_argument(
        "--request-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help="The compression algorithm to be used when sending request body to server. Default is None.",
    )
    parser.add_argument(
        "--response-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help="The compression algorithm to be used when receiving response body from server. Default is None.",
    )

    FLAGS = parser.parse_args()
    try:
        if FLAGS.ssl:
            ssl_options = {}
            if FLAGS.key_file is not None:
                ssl_options["keyfile"] = FLAGS.key_file
            if FLAGS.cert_file is not None:
                ssl_options["certfile"] = FLAGS.cert_file
            if FLAGS.ca_certs is not None:
                ssl_options["ca_certs"] = FLAGS.ca_certs
            ssl_context_factory = None
            if FLAGS.insecure:
                ssl_context_factory = gevent.ssl._create_unverified_context
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=True,
                ssl_options=ssl_options,
                insecure=FLAGS.insecure,
                ssl_context_factory=ssl_context_factory,
            )
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "scikit_learn_model"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    x_data = np.array([[-1.6685316675305422, -1.2990134593088984, 0.27464720361244455, -0.6036204360190907]],
                      dtype=np.float64)

    if FLAGS.http_headers is not None:
        headers_dict = {l.split(":")[0]: l.split(":")[1] for l in FLAGS.http_headers}
    else:
        headers_dict = None

    # Infer with requested Outputs
    results = test_infer(
        model_name,
        x_data,
        headers_dict,
        FLAGS.request_compression_algorithm,
        FLAGS.response_compression_algorithm,
    )
    print(results.get_response())

    statistics = triton_client.get_inference_statistics(
        model_name=model_name, headers=headers_dict
    )
    print(statistics)
    if len(statistics["model_stats"]) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Validate the results by comparing with precomputed values.
    label_data = results.as_numpy("label")
    probabilities_data = results.as_numpy("probabilities")

    # Infer without requested Outputs
    results = test_infer_no_outputs(
        model_name,
        x_data,
        headers_dict,
        FLAGS.request_compression_algorithm,
        FLAGS.response_compression_algorithm,
    )
    print(results.get_response())

    # Validate the results by comparing with precomputed values.
    label_data = results.as_numpy("label")
    probabilities_data = results.as_numpy("probabilities")

    # Infer with incorrect model name
    try:
        _ = test_infer("wrong_model_name", x_data).get_response()
        print("expected error message for wrong model name")
        sys.exit(1)
    except InferenceServerException as ex:
        print(ex)
        if not (ex.message().startswith("Request for unknown model")):
            print("improper error message for wrong model name")
            sys.exit(1)