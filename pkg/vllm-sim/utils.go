/*
Copyright 2025 The vLLM-Sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vllmsim

import (
	"math/rand"
	"time"
)

// list of responses to use in random mode for comepltion requests
var chatCompletionFakeResponses = []string{
	`This is a test.`,
	`The quick brown fox jumps over the lazy dog.`,
	`Lorem ipsum dolor sit amet, consectetur adipiscing elit.`,
	`To be or not to be, that is the question.`,
	`All your base are belong to us.`,
	`I am the bone of my sword.`,
	`I am the master of my fate.`,
	`I am the captain of my soul.`,
	`I am the master of my fate, I am the captain of my soul.`,
	`I am the bone of my sword, steel is my body, and fire is my blood.`,
	`The quick brown fox jumps over the lazy dog.`,
	`Lorem ipsum dolor sit amet, consectetur adipiscing elit.`,
	`To be or not to be, that is the question.`,
	`All your base are belong to us.`,
	`Omae wa mou shindeiru.`,
	`Nani?`,
	`I am inevitable.`,
	`May the Force be with you.`,
	`Houston, we have a problem.`,
	`I'll be back.`,
	`You can't handle the truth!`,
	`Here's looking at you, kid.`,
	`Go ahead, make my day.`,
	`I see dead people.`,
	`Hasta la vista, baby.`,
	`You're gonna need a bigger boat.`,
	`E.T. phone home.`,
	`I feel the need - the need for speed.`,
	`I'm king of the world!`,
	`Show me the money!`,
	`You had me at hello.`,
	`I'm the king of the world!`,
	`To infinity and beyond!`,
	`You're a wizard, Harry.`,
	`I solemnly swear that I am up to no good.`,
	`Mischief managed.`,
	`Expecto Patronum!`,
}

// getRandomResponseText returns random response text from the pre-defined list of responses
func getRandomResponseText() string {
	index := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(chatCompletionFakeResponses))
	return chatCompletionFakeResponses[index]
}
