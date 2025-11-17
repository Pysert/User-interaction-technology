<script setup>
import ChatView from './views/ChatView.vue'
</script>

<template>
  <ChatView />
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background: #f5f5f5;
  overflow-x: hidden;
}

#app {
  height: 100vh;
  width: 100%;
  overflow-x: hidden;
}
</style>
