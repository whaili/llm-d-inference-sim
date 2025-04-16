package vllmsim_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestVllmSim(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "VllmSim Suite")
}
