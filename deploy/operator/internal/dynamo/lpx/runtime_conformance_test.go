/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// The fixture is shared with lpu-monorepo/Groq/IE/rust/lpx-runtime/tests/fixtures/partitions.json.
// It compares the operator's selected placement with the manifest resolver's contract.
func TestRuntimeDescriptorConformance(t *testing.T) {
	t.Parallel()
	t.Log("Read the shared native-runtime conformance cases")
	data, err := os.ReadFile("testdata/runtime-partitions.json")
	require.NoError(t, err)
	var cases []struct {
		Name      string `json:"name"`
		Artifacts []struct {
			ID      int     `json:"id"`
			Path    string  `json:"path"`
			Chips   int     `json:"chips"`
			HXShape []int64 `json:"hx_shape"`
			Devices int     `json:"devices"`
		} `json:"artifacts"`
		Chains   [][]int `json:"chains"`
		Selected []int   `json:"selected"`
		Hybrid   bool    `json:"hybrid"`
		Expected []struct {
			ID    int    `json:"source_partition_id"`
			Path  string `json:"relative_path"`
			Nodes int    `json:"node_count"`
		} `json:"expected"`
		Error string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(data, &cases))
	for _, test := range cases {
		if test.Error != "" {
			continue
		}
		t.Run(test.Name, func(t *testing.T) {
			t.Log("Construct physical compiler partitions using the shared geometry")
			build := Build{Family: BuildFamilyXT}
			for _, artifact := range test.Artifacts {
				topology := Topology{ChipCount: artifact.Chips, Raw: "fixture"}
				part := BuildPartition{SourcePartitionID: artifact.ID, PartPath: artifact.Path, Topology: topology, DevicesPerNode: artifact.Devices, HXExtent: artifact.HXShape}
				if artifact.Devices == 16 {
					build.Family = BuildFamilyHX
					if len(part.HXExtent) == 0 {
						part.HXExtent = []int64{16, 1, 1, 1}
					}
				}
				build.Partitions = append(build.Partitions, part)
			}

			t.Log("Compare selected descriptors after the operator's existing chain collapse")
			if test.Hybrid && build.Family == BuildFamilyXT {
				build.SelectedPropSyncChains = test.Chains
				projected, err := appendV2ModelProjections(nil, ModelProjectionInput{
					Pipeline: PipelineLPX, Models: []string{"default"},
					BuildSnapshot: NormalizedBuildSnapshot{build: &build},
				})
				require.NoError(t, err)
				build = projected[0].configuredBuild
			}
			selected := Build{Family: build.Family}
			for _, id := range test.Selected {
				index := slices.IndexFunc(build.Partitions, func(part BuildPartition) bool { return part.SourcePartitionID == id })
				require.NotEqual(t, -1, index)
				selected.Partitions = append(selected.Partitions, build.Partitions[index])
			}
			projection := &ModelProjection{configuredBuild: selected, pipeline: PipelineLPX}
			var ids []string
			require.Len(t, selected.Partitions, len(test.Expected))
			for index, expected := range test.Expected {
				part := selected.Partitions[index]
				ids = append(ids, strconv.Itoa(expected.ID))
				require.Equal(t, expected.ID, part.SourcePartitionID)
				require.Equal(t, expected.Path, part.PartPath)
				nodes := part.effectiveNodeCount()
				if build.Family == BuildFamilyHX {
					nodes = int(part.HXExtent[1] * part.HXExtent[2] * part.HXExtent[3])
				}
				require.Equal(t, expected.Nodes, nodes)
			}
			require.Equal(t, strings.Join(ids, ","), runtimePartitionIDs(projection))

		})
	}
}
