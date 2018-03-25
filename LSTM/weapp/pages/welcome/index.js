//index.js
//获取应用实例
const app = getApp()

Page({
  data: {
    //motto: '测试测试！',
    userInfo: {},
    hasUserInfo: false,
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    containerShowflag: false,
    userinfoShowflag: false,
    bonjorShowflag: false,
    dianwoShowflag: false,
    poemShowflag: false,
    poemreshowflag: false,
    waitmsg: '请稍后。。。',
    status: 'waiting',
    animationData1: {},
    animationData2: {},
    animationData3: {},
    animationData4: {},
    poetrycontent: [],
    templine: [],
    linenum: 0
  }, 
  //事件处理函数
  bindViewTap: function() {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },

  onShow: function () {
    this.setData({ containerShowflag: true })
  //动画1，先显示用户信息
    this.setData({userinfoShowflag: true})
  //创建动画1
    var animation = wx.createAnimation({
      duration: 2000,
      timingFunction: 'ease'
      //transformOrigin: "10% 10%"
    })
  //将用户信息缩小到原来的0.01倍
    animation.opacity(0.01, 0.01).step()
  //触发之
    this.setData({
      animationData1: animation.export()
    })
  //设置定时器，将其还原大小
    setTimeout(function () {
      animation.opacity(1, 1).step()
      this.setData({
        animationData1: animation.export()
      })
    }.bind(this), 500)//这里的this是40行的还是本函数里面的？

    this.setData({ bonjorShowflag: true })

    let animation2 = wx.createAnimation({
      duration: 500,
      timingFunction: "ease",
      delay: 0
    });

    animation2.scale(0.01, 0.01).step();

    this.setData({
      animationData2: animation2.export()
    });

    setTimeout(function () {
      animation2.scale(1, 1).step()
      this.setData({
        animationData2: animation2.export()
      })
    }.bind(this), 1500)

    this.setData({ dianwoShowflag: true })

    let animation3 = wx.createAnimation({
      duration: 3000,
      timingFunction: "ease",
      delay: 0
    });

    animation3.opacity(0.01, 0.01).step()

    this.setData({
      animationData3: animation3.export()
    });

    setTimeout(function () {
      animation3.opacity(1, 1).step()
      this.setData({
        animationData3: animation3.export()
      })
    }.bind(this), 2000)
  },
  
  onLoad: function () {
    if (app.globalData.userInfo) {
      this.setData({
        userInfo: app.globalData.userInfo,
        hasUserInfo: true
      })
    } else if (this.data.canIUse){
      // 由于 getUserInfo 是网络请求，可能会在 Page.onLoad 之后才返回
      // 所以此处加入 callback 以防止这种情况
      app.userInfoReadyCallback = res => {
        this.setData({
          userInfo: res.userInfo,
          hasUserInfo: true
        })
      }
    } else {
      // 在没有 open-type=getUserInfo 版本的兼容处理
      wx.getUserInfo({
        success: res => {
          app.globalData.userInfo = res.userInfo
          this.setData({
            userInfo: res.userInfo,
            hasUserInfo: true
          })
        }
      })
    };
    var animation = wx.createAnimation({
      duration: 1000,
      timingFunction: 'ease',
    })
  },

  getUserInfo: function(e) {
    console.log(e)
    app.globalData.userInfo = e.detail.userInfo
    this.setData({
      userInfo: e.detail.userInfo,
      hasUserInfo: true
    })
  },
/*
  downloadPoetry: function () {
    var self = this
    wx.downloadFile({
      url: 'http://16644679.peitaiyi.xyz',
      success: function (res) {
        console.log('downloadFile success, res is', res)

        self.setData({
          poetrySrc: res.tempFilePath
        })
      },
      fail: function ({ errMsg }) {
        console.log('downloadFile fail, err is:', errMsg)
      }
    })
  },
*/
  clickMe: function () {
    /*
    wx.navigateTo({
      url: '../gen/zimu'
    });
    */
    this.setData({
      containerShowflag:false,
      userinfoShowflag: false,
      bonjorShowflag: false,
      dianwoShowflag: false,
      poemShowflag: true,
      poemreshowflag: false,
      linenum: 0
    });
    var that =this;
    wx.request({
      url: 'https://16644679.peitaiyi.xyz/getpoetry' + Math.floor(Math.random()*10), 
      /*url: 'https://16644679.peitaiyi.xyz/getpoetry0_split',*/
      method: 'GET',
      success: (res) => {
        if (+res.statusCode == 200) {
          console.log('服务器响应成功')
          this.setData({ waitmsg: '服务器响应成功!', status: 'success', poetrycontent: res.data})
        }
        else {
          this.setData({ waitmsg: '服务器无响应!', status: 'waiting' })
        }
      }

    });

    var cout = 0;
    var i = setInterval(function () {
      that.setData({ linenum:cout})
      //console.log(cout)
      cout = cout + 1;
    }, 2000)//循环时间 这里是1秒  

    setTimeout(function() {
      clearInterval(i)
      that.setData({ cout: 0 });
    },10000)

    let animation4 = wx.createAnimation({
      duration: 2000,
      timingFunction: "ease",
      delay: 0
    });

    that.setData({ poemreshowflag: true })
    animation4.opacity(0.01, 0.01).step()

    that.setData({
      animationData4: animation4.export()
    });

    setTimeout(function () {
      animation4.opacity(1, 1).step()
      that.setData({
        animationData4: animation4.export()
      })
    }.bind(that), 1000)

/*
    wx.downloadFile({
      url: 'https://16644679.peitaiyi.xyz/getpoetry',
      success: function (res) {
        console.log('downloadFile success, res is', res)
        if (res.statusCode === 200) {
          poetrySrc: res.tempFilePath
        }
      },
      fail: function ({ errMsg }) {
        console.log('downloadFile fail, err is:', errMsg)
      }
    })*/
  },

  clickMe2: function () {
    this.setData({
      containerShowflag: false,
      userinfoShowflag: false,
      bonjorShowflag: false,
      dianwoShowflag: false,
      poemShowflag: true
    });
  },

  onHide: function () {
    this.setData({ waitmsg: '请稍后...!', 
      status: 'waiting', containerShowflag:false, userinfoShowflag: false, bonjorShowflag: false, dianwoShowflag: false,poemShowflag: false})
  }

})
